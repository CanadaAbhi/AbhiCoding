# include <stdio.h>
int count = 0;

main()
{
	int vcount=0;
	int id;
	id = vfork();
	if(id == 0){
		printf("In child\n");
		++count;
		++vcount;
		printf("count    %d, vcount%d\n\n",count,vcount);
	}
	else{
		printf("In parent\n");
		++ count;
		++vcount;
		printf("count    %d, vcount%d\n\n",count,vcount);
}
}
