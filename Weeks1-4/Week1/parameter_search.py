#!/usr/bin/env python3
"""W&B Sweep parameter search (grid/random) for background subtraction."""

import argparse
import subprocess
import sys
from pathlib import Path
from multiprocessing import Process
import yaml
import wandb

SWEEP_CONFIGS = {
    'gaussian': Path(__file__).parent / 'sweep_gaussian.yaml',
    'mog2': Path(__file__).parent / 'sweep_mog2.yaml',
    'lsbp': Path(__file__).parent / 'sweep_lsbp.yaml',
    'subsense': Path(__file__).parent / 'sweep_subsense.yaml',
    'lobster': Path(__file__).parent / 'sweep_lobster.yaml',
}

def run_trial(params):
    """Run trial and log metrics to W&B."""
    cmd = ["python", "Week1/main.py"]
    for k, v in params.items():
        # Handle boolean flags that use action="store_true"
        if isinstance(v, bool) and k == 'adaptive':
            if v:  # Only add flag if True
                cmd.append(f"--{k}")
        else:
            if k == "mog2-history":
                v = int(v)  # Ensure integer for history parameter
            cmd.append(f"--{k}")
            cmd.append(str(v))
    print(f"Running trial with params: {cmd}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    
    if result.returncode != 0:
        print(f"Failed: {result.stderr}")
        return
    
    metrics = {}
    for line in result.stdout.split('\n'):
        if 'mAP@0.5:' in line: metrics['mAP@0.5'] = float(line.split(':')[1].strip())
        elif 'Recall:' in line: metrics['recall'] = float(line.split(':')[1].strip())
        elif 'Precision:' in line: metrics['precision'] = float(line.split(':')[1].strip())
        elif 'F1 Score:' in line: metrics['f1_score'] = float(line.split(':')[1].strip())
    
    if metrics:
        wandb.log(metrics)

def main():
    parser = argparse.ArgumentParser(description="W&B Sweep for parameter search")
    parser.add_argument('--method', default='gaussian', choices=['gaussian', 'mog2', 'lsbp', 'subsense', 'lobster'])
    parser.add_argument('--config', help='Custom YAML config')
    parser.add_argument('--num-samples', type=int, default=10, 
                        help='Number of trials (only for random search, ignored for grid)')
    parser.add_argument('--num-agents', type=int, default=1,
                        help='Number of parallel agents to run')
    parser.add_argument('--project', default='param-search')
    parser.add_argument('--entity', default=None)
    parser.add_argument('--sweep-id', help='Join existing sweep')
    args = parser.parse_args()

    yaml_path = Path(args.config) if args.config else SWEEP_CONFIGS[args.method]
    if not yaml_path.exists():
        sys.exit(f"Config not found: {yaml_path}")

    with open(yaml_path) as f:
        sweep_config = yaml.safe_load(f)

    if args.sweep_id:
        print(f"Joining sweep {args.sweep_id}")
        sweep_id = args.sweep_id
    else:
        sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
        print(f"Sweep created: {sweep_id}")

    # Determine count based on search method
    is_grid = sweep_config.get('method') == 'grid'
    count = None if is_grid else args.num_samples
    
    if is_grid:
        print(f"Grid search: running until all combinations are explored")
    else:
        print(f"Random search: running {args.num_samples} trials")

    def run_agent(agent_id):
        """Run a single agent."""
        def agent_fn():
            wandb.init()
            run_trial(dict(wandb.config))
            wandb.finish()
        
        wandb.agent(sweep_id, function=agent_fn, count=count, project=args.project, entity=args.entity)
        print(f"Agent {agent_id} completed")

    if args.num_agents == 1:
        run_agent(0)
    else:
        print(f"Launching {args.num_agents} parallel agents")
        processes = [Process(target=run_agent, args=(i,)) for i in range(args.num_agents)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

if __name__ == '__main__':
    main()
