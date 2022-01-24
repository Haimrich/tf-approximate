const int n = 3/*Insert N here: how many loops do you need?*/;
int i[n+1]; // if "n" is not known before hand, then this array will need to be created dynamically.
//Note: there is an extra element at the end of the array, in order to keep track of whether to exit the array.

for (int a=0; a<n+1; a++) {
  i[a]=0;
}

//That's just an example, if all of the loops are identical: e.g. "for(int i=0; i<79; i++)". 
// If the value of MAX changes for each loop, 
// then make MAX an array instead: (new) int MAX [n]; MAX[0]=10; MAX[1]=20;...;MAX[n-1]=whatever.
int MAX = 79; 

//Used to increment all of the indicies correctly, at the end of each loop.
int p = 0; 

//Remember, you're only using indicies i[0], ..., i[n-1]. The (n+1)th index, i[n], 
// is just to check whether to the nested loop stuff has finished.
while (i[n]==0) {
    

  //DO STUFF HERE. Pretend you're inside your nested for loops. The more usual i,j,k,... have been replaced here with i[0], i[1], ..., i[n-1].


  //Now, after you've done your stuff, we need to increment all of the indicies correctly.
  i[0]++;
  
  //(or "MAX[p]" if each "for" loop is different. 
  // Note that from an English point of view, this is more like "if(i[p]==MAX".
  // (Initially i[0]) If this is true, then i[p] is reset to 0, and i[p+1] is incremented.
  while(i[p]==MAX) {
    i[p]=0;
    i[++p]++; //increase p by 1, and increase the next (p+1)th index
    if(i[p]!=MAX)
      p=0;//Alternatively, "p=0" can be inserted above (currently commented-out). This one's more efficient though, since it only resets p when it actually needs to be reset!
  }
}