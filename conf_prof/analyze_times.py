'''
NOTE
Will be a script to narrow down on the possible selection of configs to be the fastest, given the set of json files.

So, I want to get the best times for all sweeps across attn_types (and of course when using gqa would liek to get the best across all n_groups)

'''

import json
import re
from tqdm import tqdm
import os

def safe_load_json(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    last_brace_pos = content.rfind('}')
    if last_brace_pos != -1:
        content = content[:last_brace_pos + 1]
        if content.endswith('}}'):
            content = content[:-1]
        elif content.endswith('}\n}'):
            content = content[:-2]
        elif re.search(r'\}-?\d+\n\}$', content):
            content = re.sub(r'\}-?\d+\n\}$', '}', content)
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        print(content)
        return None

def get_results(root_dir: str):
    results = []
    
    list_files = sorted(os.listdir(root_dir), key=lambda x: (x.split('_')[0], int(x.split('_')[2])))
    for i, file in enumerate(list_files):
        list_files[i] = os.path.join(root_dir, file, 'metrics.json')
    i = 0
    for file_path in tqdm(list_files, total=len(list_files), desc="Fetching sweep results"):
        if file_path.endswith('.json'):
            print(file_path)
            result = safe_load_json(file_path)
            results.append(result)
        i += 1

    return results

def get_best_configs(results, metric="avg_fwd_bwd_time", group_by="attn_type", top_n=None):
    best_configs = {}
    group_values = set(cfg["config"].get(group_by, None) for cfg in results)

    for group_val in group_values:
        filtered_configs = [cfg for cfg in results if cfg["config"].get(group_by) == group_val]
        if group_by == "attn_type" and group_val == "gqa":
            gqa_grouped = {}
            for cfg in filtered_configs:
                n_groups = cfg["config"].get("n_groups", None)
                gqa_grouped.setdefault(n_groups, []).append(cfg)
            best_gqa_configs = []
            for n_groups_val, cfgs in gqa_grouped.items():
                sorted_cfgs = sorted(cfgs, key=lambda x: x[metric])
                if top_n:
                    sorted_cfgs = sorted_cfgs[:top_n]
                best_gqa_configs.extend(sorted_cfgs)

            best_configs[group_val] = best_gqa_configs
        else:
            sorted_configs = sorted(filtered_configs, key=lambda x: x[metric])
            if top_n:
                sorted_configs = sorted_configs[:top_n]
            best_configs[group_val] = sorted_configs
    return best_configs

def write_results(best_configs, root_dir='conf_prof/results', file_name='time_results.md', group_by = "attn_type", top_n = 2, metric = 'avg_fwd_bwd_time'):
    os.makedirs(root_dir, exist_ok=True)
    file_path = os.path.join(root_dir, file_name)
    
    with open(file_path, 'w') as f:
        f.write(f"These are the best top-{top_n} configuration(s) across {group_by}, with respect to {metric}.\n\n")
        for group, configs in best_configs.items():
            f.write(f"# Top-{top_n} configurations for {group}\n\n")
            for config in configs:
                '''if group == 'gqa':
                    f.write(f"Number of Groups: `{config[0]}`\n\n")
                    f.write(f"Avg. Forward Time: `{config[1]['avg_forward_time']}`\n\n")
                    f.write(f"Avg. Backward Time: `{config[1]['avg_backward_time']}`\n\n")
                    f.write(f"Avg. Forward $\\rightarrow$ Backward Time: `{config[1]['avg_fwd_bwd_time']}`\n\n")
                    config_str = json.dumps(config[1], indent=2)
                    f.write(f"```json\n{config_str}\n```\n\n")
                    f.write(f"---\n\n")
                else: '''
                if group == 'gqa':
                    f.write(f"Number of Groups: `{config['config']['n_groups']}`\n\n")
                f.write(f"Avg. Forward Time: `{config['avg_forward_time']}`\n\n")
                f.write(f"Avg. Backward Time: `{config['avg_backward_time']}`\n\n")
                f.write(f"Avg. Forward $\\rightarrow$ Backward Time: `{config['avg_fwd_bwd_time']}`\n\n")
                config_str = json.dumps(config, indent=2)
                f.write(f"```json\n{config_str}\n```\n\n")
                f.write(f"---\n\n")

if __name__ == "__main__":
    root_dir = "conf_prof/results/results_json"
    top_n = 1

    results = get_results(root_dir)
    print(len(results)) 
    best_configs = get_best_configs(results, metric = "avg_fwd_bwd_time", group_by = 'attn_type', top_n = top_n)
    write_results(best_configs, root_dir = "conf_prof/results", file_name = "time_results.md", top_n = top_n)