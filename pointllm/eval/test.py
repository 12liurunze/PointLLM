import os
import re
import json
import csv
import argparse

def extract_ids(data_path):
    # 匹配 xxx_8192.npy -> xxx
    pat = re.compile(r"^(?P<oid>[0-9a-fA-F]+)_(?P<n>\d+)\.npy$")
    ids = []
    for fn in os.listdir(data_path):
        m = pat.match(fn)
        if m:
            ids.append(m.group("oid"))
    return sorted(set(ids))

def load_meta(meta_json):
    """
    支持两种常见格式：
    1) list[{"object_id": "...", "object_name": "..."}]
    2) dict[object_id] = {"name": "..."} / "name"
    """
    obj = json.load(open(meta_json, "r", encoding="utf-8"))
    mapping = {}

    if isinstance(obj, list):
        for x in obj:
            if not isinstance(x, dict):
                continue
            oid = x.get("object_id") or x.get("id") or x.get("uid")
            name = x.get("object_name") or x.get("name") or x.get("caption")
            if oid:
                mapping[str(oid)] = "" if name is None else str(name)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                name = v.get("object_name") or v.get("name") or v.get("caption")
                mapping[str(k)] = "" if name is None else str(name)
            else:
                mapping[str(k)] = "" if v is None else str(v)

    return mapping

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="/root/autodl-tmp/pointLLM/data/objaverse_data")
    ap.add_argument("--meta_json", required=True, help="/root/autodl-tmp/pointLLM/data/anno_data/PointLLM_complex_instruction_70K.json")
    ap.add_argument("--out_csv", default="/root/autodl-tmp/pointLLM/data/id_name_map.csv")
    args = ap.parse_args()

    ids = extract_ids(args.data_path)
    m = load_meta(args.meta_json)

    matched = 0
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["object_id", "name"])
        for oid in ids:
            name = m.get(oid, "")
            if name:
                matched += 1
            w.writerow([oid, name])

    print(f"total_ids={len(ids)}, matched_names={matched}, out={args.out_csv}")

if __name__ == "__main__":
    main()