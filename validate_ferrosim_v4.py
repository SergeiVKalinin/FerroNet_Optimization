"""Execute all six FerroSim v4 notebooks in isolated, small Anaconda/Jupyter kernels.

Usage (with NumPy, pandas, SciPy, Matplotlib, Numba, PyTorch, nbformat and
nbclient installed):
    python validate_ferrosim_v4.py --mode initialization
    python validate_ferrosim_v4.py --mode one-generation --approach Legendre

Each test changes ONLY a transient, in-memory copy of the notebook configuration
and executes it in a newly created temporary working directory. It never
modifies the tracked notebooks or physical results in the repository.

A passing initialization test is not a validated full research run.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import re
import tempfile
import traceback
import time
from nbclient import NotebookClient
import nbformat

NOTEBOOKS={
    "Legendre":"Legendre/FerroSim_Legendre_v4.ipynb",
    "Fourier":"Fourier/FerroSim_Fourier_v4.ipynb",
    "Spline":"Spline/FerroSim_Spline_v4.ipynb",
    "Segments":"Segments/FerroSim_Segments_v4.ipynb",
    "Grammar":"Grammar/FerroSim_Grammar_v4.ipynb",
    "Direct_Waveform":"Direct_Waveform/FerroSim_Direct_Waveform_v4.ipynb",
}
CONTROLS={
    "MODE":'"QUICK"',
    "N_DISORDERS":"1",
    "SEEDS":"[0]",
    "INITIAL_CONTROLS":"2",
    "ANALYZE_EACH_DISORDER":"True",
    "RUN_CROSS_DISORDER_TRANSFER":"False",
    "RUN_HIGHER_FIDELITY_VALIDATION":"False",
    "VALIDATE_ALL_DISORDERS":"False",
    "RUN_TEXTURE_ANALYSIS":"True",
    "TEXTURE_PLOT_EACH_DISORDER":"True",
    "TEXTURE_MAX_SNAPSHOTS":"3",
    "TEXTURE_PCA_REFERENCE_ONLY":"True",
}
def alter_config(source,generations):
    values={**CONTROLS,"GENERATIONS":str(generations)}
    for name,value in values.items():
        pattern=r"^("+re.escape(name)+r"\s*=\s*)(.*?)(\s*(?:#.*)?)$"
        new,changed=re.subn(pattern,lambda match:match.group(1)+value+match.group(3),
                            source,count=1,flags=re.MULTILINE)
        if changed!=1:raise ValueError("Expected exactly one configuration: "+name)
        source=new
    return source

def validate(path,workdir,generations,timeout):
    notebook=nbformat.read(path,as_version=4)
    nbformat.validate(notebook)
    if len(notebook.cells)<3 or notebook.cells[1].cell_type!="code":
        raise ValueError("Expected first code cell at notebook index 1")
    notebook.cells[1].source=alter_config(notebook.cells[1].source,generations)
    # All %%writefile calls rely on cwd; use the isolated test working directory.
    client=NotebookClient(notebook,timeout=timeout,kernel_name="python3",
        resources={"metadata":{"path":str(workdir)}},allow_errors=False)
    client.execute()
    exports=list(workdir.glob("FerroSim_*_v4/experiments/*/texture/texture_generation_index.csv"))
    if not exports:
        raise AssertionError("Texture generation archive was not produced")
    # Entire notebook includes the Legendre Try statistical reports.
    statistics=list(workdir.glob("FerroSim_*_v4/experiments/*/tables/paired_algorithm_differences.csv"))
    if not statistics:
        raise AssertionError("Paired statistical analysis table is missing")
    return {"texture_indexes":[str(p) for p in exports],
            "paired_statistic_tables":[str(p) for p in statistics]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["initialization","one-generation"],default="initialization")
    p.add_argument("--approach",choices=["all",*NOTEBOOKS],default="all")
    p.add_argument("--timeout",type=int,default=1800)
    p.add_argument("--output",default="ferrosim_v4_validation.json")
    args=p.parse_args()
    root=pathlib.Path(__file__).resolve().parent
    gens=0 if args.mode=="initialization" else 1
    selected=NOTEBOOKS if args.approach=="all" else {args.approach:NOTEBOOKS[args.approach]}
    logs=[]
    for approach,relative in selected.items():
        path=root/relative
        started=time.monotonic()
        print("Running",approach,"| generations",gens,flush=True)
        with tempfile.TemporaryDirectory(prefix="ferrosim_v4_"+approach.lower()+"_") as t:
            try:
                exports=validate(path,pathlib.Path(t),gens,args.timeout)
                logs.append({"approach":approach,"passed":True,
                  "seconds":time.monotonic()-started,"files":exports})
                print("PASS",approach,flush=True)
            except Exception as exc:
                logs.append({"approach":approach,"passed":False,
                  "seconds":time.monotonic()-started,"error":repr(exc),
                  "traceback":traceback.format_exc()})
                print("FAIL",approach,repr(exc),flush=True)
    pathlib.Path(args.output).write_text(json.dumps(logs,indent=2,default=str))
    print("Validation summary:",args.output,flush=True)
    if any(not row["passed"] for row in logs):raise SystemExit(1)

if __name__=="__main__":
    main()
