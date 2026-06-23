import argparse
import csv
import gc
import json
import os
import time
from pathlib import Path

import torch

from eagle_eye.evaluation.compare_pointllm_eagle import (
    build_prompt,
    decode_new_tokens,
    load_eagle_model,
    load_point_cloud,
    load_pointllm_helpers,
    summarize,
)


def parse_csv(value, cast):
    return [cast(item) for item in value.split(',') if item != '']


def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def run_eagle_quiet(args, model, tokenizer, baseline_results, helpers):
    results = []
    skipped = []
    for idx, sample in enumerate(baseline_results):
        try:
            point_clouds = load_point_cloud(args, helpers, model.base_model, sample)
            input_ids, stop_str, stopping_criteria = build_prompt(
                model.base_model, args.base_model_path, tokenizer, helpers, sample['question']
            )
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            output_ids = model.eagenerate(
                input_ids=input_ids,
                point_clouds=point_clouds,
                stopping_criteria=[stopping_criteria],
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                max_length=args.max_length,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            input_len = input_ids.shape[1]
            answer = decode_new_tokens(tokenizer, output_ids, input_len, stop_str)
            result = {
                **sample,
                'eagle_answer': answer,
                'eagle_time': elapsed,
                'eagle_new_tokens': int(output_ids.shape[1] - input_len),
            }
            result['speedup'] = result['baseline_time'] / elapsed if elapsed > 0 else None
            results.append(result)
        except Exception as exc:
            skipped.append({**sample, 'stage': 'eagle', 'error': f'{type(exc).__name__}: {exc}'})
        if (idx + 1) % args.progress_every == 0:
            print(f'[PROGRESS] combo_sample {idx + 1}/{len(baseline_results)}', flush=True)
    return results, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model-path', default='/root/autodl-tmp/point7B_v1.1')
    parser.add_argument('--pointllm-repo-path', default='/root/autodl-tmp/pointLLM')
    parser.add_argument('--data-path', default='/root/autodl-tmp/pointLLM/data/objaverse_data')
    parser.add_argument('--baseline-jsonl', required=True)
    parser.add_argument('--baseline-skipped-jsonl', default=None)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--heads', required=True, help='name=/path,name=/path')
    parser.add_argument('--spatial-points', default='0,2048,4096,6144')
    parser.add_argument('--keep-ratios', default='0.25,0.5,0.75')
    parser.add_argument('--summary-counts', default='0,8,16')
    parser.add_argument('--text-weights', default='0.0,0.5,1.0')
    parser.add_argument('--point-backbone-config-name', default='PointTransformer_8192point_2layer')
    parser.add_argument('--force-single-point-proj', action='store_true')
    parser.add_argument('--pointnum', type=int, default=8192)
    parser.add_argument('--torch-dtype', default='float16', choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=0.0)
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--progress-every', type=int, default=25)
    parser.add_argument('--device-map', default=None)
    args = parser.parse_args()

    dtype_mapping = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
    args.torch_dtype = dtype_mapping[args.torch_dtype]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_csv = outdir / 'sweep_summary.csv'
    best_json = outdir / 'best_so_far.json'

    baseline_results = load_jsonl(args.baseline_jsonl)
    baseline_skipped = load_jsonl(args.baseline_skipped_jsonl) if args.baseline_skipped_jsonl else []
    helpers = load_pointllm_helpers(args.pointllm_repo_path)

    heads = []
    for spec in args.heads.split(','):
        name, path = spec.split('=', 1)
        heads.append((name, path))
    spatial_points = parse_csv(args.spatial_points, int)
    keep_ratios = parse_csv(args.keep_ratios, float)
    summary_counts = parse_csv(args.summary_counts, int)
    text_weights = parse_csv(args.text_weights, float)

    fieldnames = [
        'head_name', 'head_path', 'spatial_points', 'keep_ratio', 'summary_count', 'text_weight',
        'num_samples', 'num_skipped', 'baseline_total_time', 'eagle_total_time', 'speedup_by_total_time',
        'baseline_tokens_per_second', 'eagle_tokens_per_second', 'baseline_total_new_tokens',
        'eagle_total_new_tokens', 'output_jsonl', 'skipped_jsonl', 'summary_json',
    ]
    if not summary_csv.exists():
        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    best = None
    combo_index = 0
    total_combos = len(heads) * len(spatial_points) * len(keep_ratios) * len(summary_counts) * len(text_weights)
    for head_name, head_path in heads:
        print(f'[HEAD] loading {head_name} {head_path}', flush=True)
        args.ee_model_path = head_path
        model, tokenizer = load_eagle_model(args)
        for spatial in spatial_points:
            for keep in keep_ratios:
                for summary_count in summary_counts:
                    for text_weight in text_weights:
                        combo_index += 1
                        combo_name = f'{head_name}_pts{spatial}_keep{keep:g}_sum{summary_count}_tw{text_weight:g}'
                        output_jsonl = outdir / f'{combo_name}.jsonl'
                        skipped_jsonl = outdir / f'{combo_name}.jsonl.skipped'
                        summary_json = outdir / f'{combo_name}_summary.json'
                        if summary_json.exists():
                            print(f'[SKIP] {combo_index}/{total_combos} existing {combo_name}', flush=True)
                            continue

                        os.environ['POINT_SPATIAL_NUM_POINTS'] = str(spatial)
                        os.environ['POINT_SPATIAL_KEEP_RATIO'] = '1.0'
                        os.environ['POINT_SPATIAL_MIN_POINTS'] = '512'
                        os.environ['POINT_TOKEN_KEEP_RATIO'] = str(keep)
                        os.environ['POINT_TOKEN_SUMMARY_COUNT'] = str(summary_count)
                        os.environ['POINT_TOKEN_TEXT_WEIGHT'] = str(text_weight)
                        os.environ.pop('DISABLE_POINT_SPATIAL_COMPRESSION', None)
                        os.environ.pop('DISABLE_POINT_TOKEN_COMPRESSION', None)

                        print(f'[COMBO] {combo_index}/{total_combos} {combo_name}', flush=True)
                        started = time.time()
                        results, eagle_skipped = run_eagle_quiet(args, model, tokenizer, baseline_results, helpers)
                        skipped = baseline_skipped + eagle_skipped
                        summary = summarize(results, skipped)
                        row = {
                            'head_name': head_name,
                            'head_path': head_path,
                            'spatial_points': spatial,
                            'keep_ratio': keep,
                            'summary_count': summary_count,
                            'text_weight': text_weight,
                            **summary,
                            'output_jsonl': str(output_jsonl),
                            'skipped_jsonl': str(skipped_jsonl),
                            'summary_json': str(summary_json),
                        }
                        write_jsonl(output_jsonl, results)
                        write_jsonl(skipped_jsonl, skipped)
                        with open(summary_json, 'w', encoding='utf-8') as f:
                            json.dump(row, f, ensure_ascii=False, indent=2)
                        with open(summary_csv, 'a', newline='', encoding='utf-8') as f:
                            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
                        if best is None or (row['speedup_by_total_time'] or 0) > (best['speedup_by_total_time'] or 0):
                            best = row
                            with open(best_json, 'w', encoding='utf-8') as f:
                                json.dump(best, f, ensure_ascii=False, indent=2)
                        print(f'[DONE] {combo_name} speedup={summary["speedup_by_total_time"]:.6f} elapsed={time.time()-started:.1f}s', flush=True)
                        if best:
                            print(f'[BEST] {best["head_name"]} pts={best["spatial_points"]} keep={best["keep_ratio"]} sum={best["summary_count"]} tw={best["text_weight"]} speedup={best["speedup_by_total_time"]:.6f}', flush=True)
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print('[SWEEP_DONE]', flush=True)


if __name__ == '__main__':
    main()
