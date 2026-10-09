"""Run the experiment grid of project outline v2.

Finished runs (metrics.json exists) are skipped, so the script can be stopped and restarted.

  E1  architectures, 12 leads                3 models x 3 seeds                         =  9
  E1b ablations, 12 leads, seed 0            plain CNN, kernel 3/15/31, vanilla RNN, pos_weight = 6
  E2  reduced leads, S1 from scratch         3 models x 5 lead sets x 3 seeds           = 45
  E3  strategies                             S2 fine-tune x 5 sets x 3 seeds + S2 frozen x lead I x 3
                                             + S3 LA-ResNet x 3 (tested on all 6 sets)   = 21
  E4  T2 triage                              ResNet1d x {12lead, leadI} x 3 + LA-ResNet x 3 = 9
  E5  label efficiency, seed 0               {10, 25, 50}% x (ResNet1d 12lead, ResNet1d leadI,
                                             S2 fine-tune leadI, LA-ResNet)              = 12
E0 (baselines) is scripts/baselines.py.

python scripts/run_experiments.py --exp e1              # one block
python scripts/run_experiments.py --exp core            # E1 + E2 + E3 + E4 (the "must" part)
python scripts/run_experiments.py --exp all             # everything
python scripts/run_experiments.py --exp e1 --seeds 0    # quick first pass
python scripts/run_experiments.py --exp all --list      # only print what would run
"""
import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from src.data import REDUCED_SETS  # noqa: E402
from src.train import parse_args, run, run_tag  # noqa: E402

ARCHS = ["resnet1d", "bilstm", "vit"]
FRACTIONS = ["0.1", "0.25", "0.5"]
BLOCKS = {"core": ["e1", "e2", "e3", "e4"], "all": ["e1", "e1b", "e2", "e3", "e4", "e5"]}


def src12(seed, out):
    return os.path.join(out, f"resnet1d_12lead_frac1_seed{seed}")


def jobs_for(block, seeds, out):
    J = []
    if block == "e1":
        J += [["--model", m, "--leads", "12lead", "--seed", str(s)] for s in seeds for m in ARCHS]
    elif block == "e1b":
        base = ["--leads", "12lead", "--seed", "0"]
        J += [["--model", "resnet1d", "--no_residual"] + base,
              ["--model", "resnet1d", "--kernel", "3"] + base,
              ["--model", "resnet1d", "--kernel", "15"] + base,
              ["--model", "resnet1d", "--kernel", "31"] + base,
              ["--model", "bilstm", "--cell", "rnn"] + base,
              ["--model", "resnet1d", "--pos_weight"] + base]
    elif block == "e2":
        J += [["--model", m, "--leads", l, "--seed", str(s)] for s in seeds for l in REDUCED_SETS for m in ARCHS]
    elif block == "e3":
        J += [["--model", "resnet1d", "--leads", l, "--seed", str(s), "--init_from", src12(s, out), "--transfer",
               "finetune"] for s in seeds for l in REDUCED_SETS]
        J += [["--model", "resnet1d", "--leads", "leadI", "--seed", str(s), "--init_from", src12(s, out),
               "--transfer", "frozen"] for s in seeds]
        J += [["--model", "laresnet", "--seed", str(s)] for s in seeds]
    elif block == "e4":
        J += [["--model", "resnet1d", "--leads", l, "--task", "t2", "--seed", str(s)] for s in seeds
              for l in ["12lead", "leadI"]]
        J += [["--model", "laresnet", "--task", "t2", "--seed", str(s)] for s in seeds]
    elif block == "e5":
        for f in FRACTIONS:
            J += [["--model", "resnet1d", "--leads", "12lead", "--train_fraction", f, "--seed", "0"],
                  ["--model", "resnet1d", "--leads", "leadI", "--train_fraction", f, "--seed", "0"],
                  ["--model", "resnet1d", "--leads", "leadI", "--train_fraction", f, "--seed", "0",
                   "--init_from", src12(0, out), "--transfer", "finetune"],
                  ["--model", "laresnet", "--train_fraction", f, "--seed", "0"]]
    else:
        raise ValueError(block)
    return J


def done_path(args):
    leads = "leadI" if args.model == "laresnet" else args.leads   # LA-ResNet writes one folder per lead set
    return os.path.join(args.out, run_tag(args, leads), "metrics.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="e1", choices=["e1", "e1b", "e2", "e3", "e4", "e5", "core", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--extra", default="", help='extra args for train, e.g. "--epochs 5 --cache data/mock-cache"')
    ap.add_argument("--list", action="store_true", help="only list the planned runs")
    a = ap.parse_args()

    extra = a.extra.split()
    out = parse_args(extra).out
    blocks = BLOCKS.get(a.exp, [a.exp])
    jobs = [(b, j) for b in blocks for j in jobs_for(b, a.seeds, out)]
    print(f"{len(jobs)} runs planned in {blocks}")
    for i, (b, j) in enumerate(jobs, 1):
        args = parse_args(j + extra)
        tag = run_tag(args, "all" if args.model == "laresnet" else args.leads)
        if os.path.exists(done_path(args)):
            print(f"[{i}/{len(jobs)}] {b} skip {tag} (done)")
            continue
        if a.list:
            print(f"[{i}/{len(jobs)}] {b} todo {tag}")
            continue
        if args.init_from and not os.path.exists(os.path.join(args.init_from, "best.pt")):
            print(f"[{i}/{len(jobs)}] {b} WAIT {tag}: needs {args.init_from} (run E1 first)")
            continue
        print(f"[{i}/{len(jobs)}] {b} run {tag}")
        run(args)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
