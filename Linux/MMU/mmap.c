/*This program illustrates how a file is mapped to a memory and written to it*/
#include<unistd.h>
#include<sys/mman.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<stdio.h>
#include<syscall.h>
#include<stdlib.h>

main(int argc,char *argv[])
{
	int fd;
	void *addr;

	if( (fd=open(argv[1],O_RDWR|O_CREAT,0777)) < 0) {
       		perror("open");
        	exit(1);
     	}

        lseek(fd,5,SEEK_SET); 
    	write(fd,"",1);
  	//lseek(fd,0,SEEK_SET);
  	addr=mmap(0,5,PROT_WRITE,MAP_SHARED,fd,0); /* Mapping the file to memory*/
	close(fd); 
	sprintf(addr,"%s\n","hello"); /* written integer 23 to the memory mapped file*/
    	munmap(addr,5); 
}
