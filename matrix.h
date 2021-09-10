#pragma once

#include "vectors.h"

struct mat4
{
	float4 row[4];

	//constructors
	explicit mat4(const float value = 0.0f)
	{
		row[0] = float4(1.0f, 0.0f, 0.0f, 0.0f);
		row[1] = float4(0.0f, 1.0f, 0.0f, 0.0f);
		row[2] = float4(0.0f, 0.0f, 1.0f, 0.0f);
		row[3] = float4(0.0f, 0.0f, 0.0f, 1.0f);
	}

	explicit mat4(float4 r0, float4 r1, float4 r2, float4 r3)
	{
		row[0] = r0;
		row[1] = r1; 
		row[2] = r2; 
		row[3] = r3; 
	}

	explicit mat4(float *p)
	{
		int index = 0;
		for(int i = 0; i < 4; i++)
		{
			row[i] = float4(p[index + 0], p[index + 1], p[index + 2], p[index + 3]);
			index += 4;
		}
	}

	//multidimensional array access operators: mat4(row, column)
	VM_INLINE float  operator() (size_t r, size_t c) const 
	{
		switch((int)c)
		{
			case 0:
			{
				return row[r].x();
				break;
			}
			case 1:
			{
				return row[r].y();
				break;
			}
			case 2:
			{
				return row[r].z();
				break;
			}
			case 3:
			{
				return row[r].w();
				break;
			}
		}
	}

	VM_INLINE float&  operator() (size_t r, size_t c) {return row[r][c];};
};

//mat4 operations
//because we're using _vectorcall, we don't need to use references/pointers

VM_INLINE mat4 operator+ (mat4 a, mat4 b) 
{
	a.row[0] = a.row[0] + b.row[0];
	a.row[1] = a.row[1] + b.row[1];
	a.row[2] = a.row[2] + b.row[2];
	a.row[3] = a.row[3] + b.row[3];
	
	return a;
}

VM_INLINE mat4 operator- (mat4 a, mat4 b) 
{
	a.row[0] = a.row[0] - b.row[0];
	a.row[1] = a.row[1] - b.row[1];
	a.row[2] = a.row[2] - b.row[2];
	a.row[3] = a.row[3] - b.row[3];
	
	return a;
}

//linear combination: row-vector * matrix
VM_INLINE float4 operator* (float4 a, mat4 b)
{
	float4 result;	
	result = a.x() * b.row[0];
	result += a.y() * b.row[1];
	result += a.z() * b.row[2];
	result += a.w() * b.row[3];
	
	return result;
}

VM_INLINE mat4 operator* (mat4 a, mat4 b) 
{
	a.row[0] = a.row[0] * b;
	a.row[1] = a.row[1] * b;
	a.row[2] = a.row[2] * b;
	a.row[3] = a.row[3] * b;
	
	return a;
}
	
VM_INLINE void transpose(mat4 &a)
{
	_MM_TRANSPOSE4_PS(a.row[0].m, a.row[1].m, a.row[2].m, a.row[3].m);
}