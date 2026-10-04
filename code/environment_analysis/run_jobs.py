from pathlib import Path
import subprocess,os,concurrent.futures,json,time
r=Path(__file__).resolve().parents[2];out=r/'logs/environment_analysis'
env=os.environ.copy();env.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
commands={'rf_oof':['/home/snakehand/Documents/mdpi_papers/.venv/bin/python','-u',str(r/'code/environment_analysis/rf_oof.py')],'occupancy':['/home/snakehand/micromamba/envs/r/bin/Rscript',str(r/'code/environment_analysis/occupancy_predictions.R')]}
def job(name,cmd):
 with (out/(name+'.log')).open('w') as f:
  t=time.time();res=subprocess.run(cmd,cwd=r.parent,env=env,stdout=f,stderr=subprocess.STDOUT)
 status={'job':name,'returncode':res.returncode,'seconds':time.time()-t};(out/(name+'_status.json')).write_text(json.dumps(status));return status
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 results=list(ex.map(lambda item:job(*item),commands.items()))
 for x in results:print(x,flush=True)
 assert all(x['returncode']==0 for x in results),results
