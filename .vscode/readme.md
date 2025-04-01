

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



du -sh /System/Volumes/Data/* | sort -h



df -h /Users/sachadrevet/data_tmp 

Filesystem     Size   Used  Avail Capacity iused      ifree %iused  Mounted on
/dev/disk3s5  460Gi  426Gi  2.4Gi   100% 4546151 4823430369    0%   /System/Volumes/Data
```


## Docker 
In settings I changed the virtual disk size limit from 64GB to 8GB.

Docker is usually causing issues:
```
du -sh /Users/sachadrevet/Library/Containers/com.docker.docker/Data 

this should remove 6GB
docker system prune -a 
```
