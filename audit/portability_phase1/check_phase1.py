from pathlib import Path
import subprocess,json,sys
D=Path('/home/abdelrahman/Hadwa/pedestrian-intent-thesis'); tool=D/'tools/evidence.py'; A=D/'audit/portability_phase1'
checks=[]
for label,args,ok in [
 ('arbitrary_cwd_integrity',['verify'],True),
 ('reject_output_outside_runs',['summaries','--output',str(D/'must_not_be_created')],False),
 ('reject_existing_output',['summaries','--output',str(D/'runs/summary_check_20261019T214049013038Z')],False),
]:
    command=[sys.executable,str(tool),*args];r=subprocess.run(command,cwd='/tmp',capture_output=True,text=True)
    assert (r.returncode==0)==ok,(label,r.stdout,r.stderr)
    checks.append({'check':label,'command':command,'cwd':'/tmp','returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
assert not (D/'must_not_be_created').exists()
(A/'command_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
(A/'commands.txt').write_text('python /tmp/continue_thesis_companion.py\npython /home/abdelrahman/Hadwa/pedestrian-intent-thesis/tools/evidence.py summaries\npython /tmp/check_thesis_companion_phase1.py\nAdditional exact subprocess commands and outcomes: command_checks.json\n')
(A/'check_phase1.py').write_bytes(Path(__file__).read_bytes())
(D/'THESIS_LINK_HANDOFF.md').write_text('''# Thesis link handoff

The intended citation destination is this new companion repository, not the original research repository. No GitHub remote or public URL exists yet. Do not insert a speculative URL.

After publication and release validation, update both `thesis/manuscript/Backmatter/ExperimentMap.tex` (Appendix A repository hyperlink) and `thesis/14_RESULT_SOURCE_MAP.md` (repository hyperlink and release-status explanation) in the thesis-writing session. Prefer the final release tag or exact commit for traceability. Retain existing experiment identifiers and update detailed file links to the final published layout. The current staging `snapshot/` layout is provisional.

Suggested wording, to fill in only after publication:

“Code and supporting experiment records are available in the thesis companion repository [repository URL], release [tag / commit]. The result-to-experiment index identifies the source files for each reported table and figure.”

Do not claim end-to-end reproducibility until the relevant raw-data, preprocessing, training, and evaluation pathways have been validated. Current evidence checks establish snapshot integrity and summary aggregation only.
''')
readme=D/'README.md';s=readme.read_text();s+='\n## First portable checks\n\nSee [REPRODUCING.md](REPRODUCING.md) for the standard-library evidence verification and summary rebuilding commands. Publication-link instructions for the separate thesis session are in [THESIS_LINK_HANDOFF.md](THESIS_LINK_HANDOFF.md). Training portability remains unfinished.\n';readme.write_text(s)
print('Three command checks passed; thesis-link handoff written. Original thesis files were not edited.')
