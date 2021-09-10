#pragma once

#include <stdlib.h>
#include <iostream>

#include "../matrix.h"

union float44
{
	float m[4][4];
	__m128 row[4];
};

//reference matrix multiplication
void float44_mul(float44 &out, float44 &A, float44 &B)
{
    float44 temp; 
    for (int i=0; i < 4; i++)
	{
        for (int j=0; j < 4; j++)
		{
            temp.m[i][j] = A.m[i][0]*B.m[0][j] + 
			               A.m[i][1]*B.m[1][j] + 
						   A.m[i][2]*B.m[2][j] + 
						   A.m[i][3]*B.m[3][j];
		}
	}
	out = temp;	
}

//reference matrix sum 
void float44_sum(float44 &out, float44 &A, float44 &B)
{
    float44 temp; 
    for (int i=0; i < 4; i++)
	{
        for (int j=0; j < 4; j++)
		{
            temp.m[i][j] = A.m[i][j] + B.m[i][j];		
		}
	}
	out = temp;	
}

//reference matrix sub 
void float44_sub(float44 &out, float44 &A, float44 &B)
{
    float44 temp; 
    for (int i=0; i < 4; i++)
	{
        for (int j=0; j < 4; j++)
		{
            temp.m[i][j] = A.m[i][j] - B.m[i][j];		
		}
	}
	out = temp;	
}

float randf() 
{
	return (rand() - 16384.0f) / 1024.0f;
}

static mat4 rand_mat4()
{
	mat4 M = mat4(0.0f);
	for(int i = 0; i < 4; i++)
		M.row[i] = float4(randf(), randf(), randf(), randf());	
	
	return M;
}

//compare mat4 implementation against reference  
void correctness_test()
{
	mat4 result_mat4;
	float44 result_ref;

	for(int i = 0; i < 1000000; i++)	
	{
		mat4 A = rand_mat4();
		mat4 B = rand_mat4();
		
		float44 C{};
		float44 D{};
		for(int i = 0; i < 4; i++)
		{
			C.row[i] = A.row[i].m;	
			D.row[i] = B.row[i].m;		
		}
		
		//multiplication test
		
		result_mat4 = A * B;
		float44_mul(result_ref, C, D);	
		if(memcmp(&result_mat4, &result_ref, sizeof(mat4)) != 0)	
		{
			std::cout << "FAILED:: Matrix Multiplication.\n";
			exit(1);
		}
		
		result_mat4 = A + B;
		float44_sum(result_ref, C, D);	
		if(memcmp(&result_mat4, &result_ref, sizeof(mat4)) != 0)	
		{
			std::cout << "FAILED:: Matrix Sum.\n";
			exit(1);
		}

		result_mat4 = A - B;
		float44_sub(result_ref, C, D);	
		if(memcmp(&result_mat4, &result_ref, sizeof(mat4)) != 0)	
		{
			std::cout << "FAILED:: Matrix Subtraction.\n";
			exit(1);
		}

	}
	std::cout << "SUCCESS :: correctness test passed.\n";
}

int main()
{
	correctness_test();
}




