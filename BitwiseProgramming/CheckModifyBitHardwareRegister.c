#include <stdio.h>
#include<stdbool.h>
#include<stdint.h>
#define REG (*(volatile uint32_t *)0x40021000)



int main()
{
  // Set bit 5 (make bit 5 = 1):  
  REG |= (1 << 5);
  // Clear bit 3 (make bit 3 = 0):  
  REG &= ~(1 << 3);
  // Toggle bit 2:  
  REG ^= (1 << 2);
  // Check if bit 7 is set:  
  bool isSet = REG & (1 << 7);


return 0;

}
