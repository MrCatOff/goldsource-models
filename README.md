# preview only, writes nothing (check the bone budget before committing)
```bash
.\.venv\Scripts\python.exe -m goldsource analyze storage/decompiled/pistols  
```
                                                                                                                                                                                                                                                                                                                   
# skip a weapon
```bash
.\.venv\Scripts\python.exe -m goldsource merge storage/decompiled/pistols -o storage/build/pistols -n v_pistols --compile --exclude v_skull1
```

# recompile an existing QC without re-merging
```bash
.\.venv\Scripts\python.exe -m goldsource compile storage/build/pistols/v_pistols.qc
```