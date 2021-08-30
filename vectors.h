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

#define SHUFFLE2(V, X,Y) float2(_mm_shuffle_ps((V).m, (V).m, _MM_SHUFFLE(Y,Y,Y,X)))
#define SHUFFLE3(V, X,Y,Z) float3(_mm_shuffle_ps((V).m, (V).m, _MM_SHUFFLE(Z,Z,Y,X)))
#define SHUFFLE4(V, X,Y,Z,W) float4(_mm_shuffle_ps((V).m, (V).m, _MM_SHUFFLE(W,Z,Y,X)))
//Examples: SHUFFLE3(v, 0,1,2) leaves the vector unchanged.
//          SHUFFLE3(v, 0,0,0) replicates the X coordinate throughout the vector.

typedef float3 bool2; //SIMD comparison operators return multiple values;
typedef float3 bool3; //SIMD comparison operators return multiple values;
typedef float4 bool4; //SIMD comparison operators return multiple values;

//-------------------- FLOAT2 ------------------------------
struct float2
{   
	__m128 m;

	VM_INLINE float2(){}	

	VM_INLINE explicit float2(const float *p){ m = _mm_set_ps(p[1], p[1], p[1], p[0]); }
	VM_INLINE explicit float2(float x, float y) { m = _mm_set_ps(y, y, y, x); }
	VM_INLINE explicit float2(__m128 v) { m = v; }

	VM_INLINE float x() const { return _mm_cvtss_f32(m); }
	VM_INLINE float y() const { return _mm_cvtss_f32(_mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 1, 1, 1))); }
	
	VM_INLINE float2 yx() const { return SHUFFLE2(*this, 1, 0); }
	VM_INLINE float2 xx() const { return SHUFFLE2(*this, 0, 0); }
	VM_INLINE float2 yy() const { return SHUFFLE2(*this, 1, 1); }

	//unaligned store	
	VM_INLINE void store(float *p) const { p[0] = x(); p[1] = y();}

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
	
	VM_INLINE float operator[] (size_t i) const { return m.m128_f32[i]; };
	VM_INLINE float& operator[] (size_t i) { return m.m128_f32[i]; };
};

VM_INLINE float2 float2i(int x, int y) { return float2((float)x, (float)y); }

//float2 operators:
VM_INLINE float2  operator+ (float2 a, float2 b) {a.m = _mm_add_ps(a.m, b.m); return a;} 
VM_INLINE float2  operator- (float2 a, float2 b) {a.m = _mm_sub_ps(a.m, b.m); return a;}
VM_INLINE float2  operator* (float2 a, float2 b) {a.m = _mm_mul_ps(a.m, b.m); return a;}; 
VM_INLINE float2  operator/ (float2 a, float2 b) {a.m = _mm_div_ps(a.m, b.m); return a;}; 
VM_INLINE float2  operator* (float2 a, float  b) {a.m = _mm_mul_ps(a.m, _mm_set1_ps(b)); return a;}; 
VM_INLINE float2  operator/ (float2 a, float  b) {a.m = _mm_div_ps(a.m, _mm_set1_ps(b)); return a;}; 
VM_INLINE float2  operator* (float  a, float2 b) {b.m = _mm_mul_ps(_mm_set1_ps(a), b.m); return b;}; 
VM_INLINE float2  operator/ (float  a, float2 b) {b.m = _mm_div_ps(_mm_set1_ps(a), b.m); return b;}; 
VM_INLINE float2& operator+= (float2 &a, float2 b) {a = a + b; return a;}
VM_INLINE float2& operator-= (float2 &a, float2 b) {a = a - b; return a;} 
VM_INLINE float2& operator*= (float2 &a, float2 b) {a = a * b; return a;}
VM_INLINE float2& operator/= (float2 &a, float2 b) {a = a / b; return a;}
VM_INLINE float2& operator*= (float2 &a, float  b) {a = a * b; return a;}
VM_INLINE float2& operator/= (float2 &a, float  b) {a = a / b; return a;}
VM_INLINE bool2   operator== (float2 a, float2 b) {a.m = _mm_cmpeq_ps(a.m, b.m); return a;}
VM_INLINE bool2   operator!= (float2 a, float2 b) {a.m = _mm_cmpneq_ps(a.m, b.m); return a;}
VM_INLINE bool2   operator< (float2 a, float2 b) {a.m = _mm_cmplt_ps(a.m, b.m); return a;}
VM_INLINE bool2   operator> (float2 a, float2 b) {a.m = _mm_cmpgt_ps(a.m, b.m); return a;}
VM_INLINE bool2   operator<= (float2 a, float2 b) {a.m = _mm_cmple_ps(a.m, b.m); return a;}
VM_INLINE bool2   operator>= (float2 a, float2 b) {a.m = _mm_cmpge_ps(a.m, b.m); return a;}
VM_INLINE float2  min(float2 a, float2 b) {a.m = _mm_min_ps(a.m, b.m); return a;}
VM_INLINE float2  max(float2 a, float2 b) {a.m = _mm_max_ps(a.m, b.m); return a;}

//-------------- FLOAT3 ----------------------
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

VM_INLINE float3 float3i(int x, int y, int z) { return float3((float)x, (float)y, (float)z);}

//float3 operators:
VM_INLINE float3  operator+ (float3 a, float3 b) {a.m = _mm_add_ps(a.m, b.m); return a;} 
VM_INLINE float3  operator- (float3 a, float3 b) {a.m = _mm_sub_ps(a.m, b.m); return a;}
VM_INLINE float3  operator* (float3 a, float3 b) {a.m = _mm_mul_ps(a.m, b.m); return a;}; 
VM_INLINE float3  operator/ (float3 a, float3 b) {a.m = _mm_div_ps(a.m, b.m); return a;}; 
VM_INLINE float3  operator* (float3 a, float  b) {a.m = _mm_mul_ps(a.m, _mm_set1_ps(b)); return a;}; 
VM_INLINE float3  operator/ (float3 a, float  b) {a.m = _mm_div_ps(a.m, _mm_set1_ps(b)); return a;}; 
VM_INLINE float3  operator* (float  a, float3 b) {b.m = _mm_mul_ps(_mm_set1_ps(a), b.m); return b;}; 
VM_INLINE float3  operator/ (float  a, float3 b) {b.m = _mm_div_ps(_mm_set1_ps(a), b.m); return b;}; 
VM_INLINE float3& operator+= (float3 &a, float3 b) {a = a + b; return a;}
VM_INLINE float3& operator-= (float3 &a, float3 b) {a = a - b; return a;} 
VM_INLINE float3& operator*= (float3 &a, float3 b) {a = a * b; return a;}
VM_INLINE float3& operator/= (float3 &a, float3 b) {a = a / b; return a;}
VM_INLINE float3& operator*= (float3 &a, float  b) {a = a * b; return a;}
VM_INLINE float3& operator/= (float3 &a, float  b) {a = a / b; return a;}
VM_INLINE bool3   operator== (float3 a, float3 b) {a.m = _mm_cmpeq_ps(a.m, b.m); return a;}
VM_INLINE bool3   operator!= (float3 a, float3 b) {a.m = _mm_cmpneq_ps(a.m, b.m); return a;}
VM_INLINE bool3   operator< (float3 a, float3 b) {a.m = _mm_cmplt_ps(a.m, b.m); return a;}
VM_INLINE bool3   operator> (float3 a, float3 b) {a.m = _mm_cmpgt_ps(a.m, b.m); return a;}
VM_INLINE bool3   operator<= (float3 a, float3 b) {a.m = _mm_cmple_ps(a.m, b.m); return a;}
VM_INLINE bool3   operator>= (float3 a, float3 b) {a.m = _mm_cmpge_ps(a.m, b.m); return a;}
VM_INLINE float3  min(float3 a, float3 b) {a.m = _mm_min_ps(a.m, b.m); return a;}
VM_INLINE float3  max(float3 a, float3 b) {a.m = _mm_max_ps(a.m, b.m); return a;}


//-------------- FLOAT4 ----------------------
struct float4
{   
	__m128 m;

	VM_INLINE float4(){}	

	VM_INLINE explicit float4(const float *p){ m = _mm_set_ps(p[3], p[2], p[1], p[0]); }
	VM_INLINE explicit float4(float x, float y, float z, float w) { m = _mm_set_ps(w, z, y, x); }
	VM_INLINE explicit float4(__m128 v) { m = v; }


	VM_INLINE float x() const { return _mm_cvtss_f32(m); }
	VM_INLINE float y() const { return _mm_cvtss_f32(_mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 1, 1, 1))); }
	VM_INLINE float z() const { return _mm_cvtss_f32(_mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 2, 2, 2))); }
	VM_INLINE float w() const { return _mm_cvtss_f32(_mm_shuffle_ps(m, m, _MM_SHUFFLE(3, 3, 3, 3))); }
	
	//TODO: add more swizzles
	VM_INLINE float4 wyzx() const { return SHUFFLE4(*this, 3, 1, 2, 0); }
	VM_INLINE float4 wzxy() const { return SHUFFLE4(*this, 3, 2, 0, 1); }
	
	VM_INLINE void store(float *p) const { p[0] = x(); p[1] = y(); p[2] = z(); p[3] = w(); }

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
	
	void setW(float w)
	{
		__m128 t = _mm_move_ss(m, _mm_set_ss(w));
		t = _mm_shuffle_ps(t, t, _MM_SHUFFLE(0, 2, 1, 0));
		m = _mm_move_ss(t, m);
	}
	
	VM_INLINE float operator[] (size_t i) const { return m.m128_f32[i]; };
	VM_INLINE float& operator[] (size_t i) { return m.m128_f32[i]; };

};

VM_INLINE float4 float4i(int x, int y, int z) { return float3((float)x, (float)y, (float)z, (float)w);}

//float4 operators:
VM_INLINE float4  operator+ (float4 a, float4 b) {a.m = _mm_add_ps(a.m, b.m); return a;} 
VM_INLINE float4  operator- (float4 a, float4 b) {a.m = _mm_sub_ps(a.m, b.m); return a;}
VM_INLINE float4  operator* (float4 a, float4 b) {a.m = _mm_mul_ps(a.m, b.m); return a;}; 
VM_INLINE float4  operator/ (float4 a, float4 b) {a.m = _mm_div_ps(a.m, b.m); return a;}; 
VM_INLINE float4  operator* (float4 a, float  b) {a.m = _mm_mul_ps(a.m, _mm_set1_ps(b)); return a;}; 
VM_INLINE float4  operator/ (float4 a, float  b) {a.m = _mm_div_ps(a.m, _mm_set1_ps(b)); return a;}; 
VM_INLINE float4  operator* (float  a, float4 b) {b.m = _mm_mul_ps(_mm_set1_ps(a), b.m); return b;}; 
VM_INLINE float4  operator/ (float  a, float4 b) {b.m = _mm_div_ps(_mm_set1_ps(a), b.m); return b;}; 
VM_INLINE float4& operator+= (float4 &a, float4 b) {a = a + b; return a;}
VM_INLINE float4& operator-= (float4 &a, float4 b) {a = a - b; return a;} 
VM_INLINE float4& operator*= (float4 &a, float4 b) {a = a * b; return a;}
VM_INLINE float4& operator/= (float4 &a, float4 b) {a = a / b; return a;}
VM_INLINE float4& operator*= (float4 &a, float  b) {a = a * b; return a;}
VM_INLINE float4& operator/= (float4 &a, float  b) {a = a / b; return a;}
VM_INLINE bool4   operator== (float4 a, float4 b) {a.m = _mm_cmpeq_ps(a.m, b.m); return a;}
VM_INLINE bool4   operator!= (float4 a, float4 b) {a.m = _mm_cmpneq_ps(a.m, b.m); return a;}
VM_INLINE bool4   operator< (float4 a, float4 b) {a.m = _mm_cmplt_ps(a.m, b.m); return a;}
VM_INLINE bool4   operator> (float4 a, float4 b) {a.m = _mm_cmpgt_ps(a.m, b.m); return a;}
VM_INLINE bool4   operator<= (float4 a, float4 b) {a.m = _mm_cmple_ps(a.m, b.m); return a;}
VM_INLINE bool4   operator>= (float4 a, float4 b) {a.m = _mm_cmpge_ps(a.m, b.m); return a;}
VM_INLINE float4  min(float4 a, float4 b) {a.m = _mm_min_ps(a.m, b.m); return a;}
VM_INLINE float4  max(float4 a, float4 b) {a.m = _mm_max_ps(a.m, b.m); return a;}

