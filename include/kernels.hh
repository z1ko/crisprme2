#pragma once


void mine(unsigned char* data, int N);

/// @brief  
/// @param query query string to match onto the strings 
/// @param trgts target strings
/// @param qlen length of the query string
/// @param tlen length of the target strings
/// @param n number of target strings
void filter(
    const unsigned char* query, 
    const unsigned char* trgts, 
    unsigned char* result, 
    int qlen, 
    int tlen, 
    int n
);