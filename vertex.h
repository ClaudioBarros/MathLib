#pragma once

#include <stdint.h>
#include <math.h>
#include <xmmintrin.h>

#define VM_INLINE     _forceinline
#define M_PI          3.14159265358979323846f
#define DEG2RAD(_a)   ((_a)*M_PI/180.0f) 
#define RAD2DEG(_a)   ((_a)*180.0f/M_PI)
#define INT_MIN       (-2147483647 - 1)
#define INT_MAX       2147483647
#define FLT_MAX       3.402823466e+38F

#define SHUFFLE3(V, X,Y,Z) float3(_mm_shuffle_ps((V).m, (V).m, _MM_SHUFFLE(Z,Z,Y,X)))

struct float3
{   
	__m128 m;

	VM_INLINE float3(){}	

	VM_INLINE explicit float3(const float *p){ m = _mm_set_ps(p[2], p[2], p[1], p[0]); }
	VM_INLINE explicit float3(float x, float y, float z) { m = _mm_set_ps(z, z, y, x); }
	VM_INLINE explicit float3(__m128 v) { m = v; }

	VM_INLINE float x() const { return _mm_cvtss_f32(m); }
	VM_INLINE float y() const { return _mm_cvtss_f32(_mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 1, 1, 1))); }
	VM_INLINE float z() const { return _mm_cvtss_f32(_mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 2, 2, 2))); }
	
	VM_INLINE float3 yzx() const { return SHUFFLE3(*this, 1, 2, 0); }
	VM_INLINE float3 zxy() const { return SHUFFLE3(*this, 2, 0, 1); }
	
	VM_INLINE void store(float *p) const { p[0] = x(); p[1] = y(); p[2] = z(); }

	void setX(float x)
	{
		m = _mm_move_ss(m, _mm_set_ss(x));
	}
	
	void setY(float y)
	{
		__m128 t = _mm_move_ss(m, _mm_set_ss(y));
		t = _mm_shuffle_ps(t, t, _MM_SHUFFLE(3, 2, 0, 0));
		m = _mm_move_ss(t, m);
	}
	
	void setZ(float z)
	{
		__m128 t = _mm_move_ss(m, _mm_set_ss(z));
		t = _mm_shuffle_ps(t, t, _MM_SHUFFLE(3, 0, 1, 0));
		m = _mm_move_ss(t, m);
	}
	
	VM_INLINE float operator[] (size_t i) const { return m.m128_f32[i]; };
	VM_INLINE float& operator[] (size_t i) { return m.m128_f32[i]; };
};

/* 
TODO:
- float2, float4, 
- operators
*/