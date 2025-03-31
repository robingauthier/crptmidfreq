

## Optimizing imports for multiple files:
 Run isort to sort and group imports in all files in a folder:
```
isort path\to\your\folder
```


Remove unused imports with autoflake:
```
autoflake --in-place --remove-all-unused-imports --ignore-init-module-imports --recursive /Users/sachadrevet/src/crptmidfreq
```



## Checking storage
```
du -sh ./* | sort -h

```