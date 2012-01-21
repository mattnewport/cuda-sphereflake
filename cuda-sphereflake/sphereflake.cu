#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <memory>
#include <limits>
#include <math.h>
#include <cassert>

using namespace std;

const unsigned int width = 640;
const unsigned int height = (width * 3) / 4;	// Maintain a 4/3 aspect ratio

const char* imageName = "sphereflake.ppm";

const float PI = 3.141592654f;

float DegToRad(float deg)
{
	return deg * ((2.0f * PI) / 360.0f);
}

class Vector3
{
public:
	__host__ __device__ void Set(const float x_, const float y_, const float z_) { x = x_; y = y_; z = z_; }

	__host__ __device__ Vector3& operator+=(const Vector3& b) { x += b.x; y += b.y; z += b.z; return *this; }

	float x;
	float y;
	float z;
};

__host__ __device__ Vector3 MakeVector3(const float x, const float y, const float z) 
{
	Vector3 v;
	v.Set(x, y, z);
	return v;
}

__host__ __device__ float Dot(const Vector3& a, const Vector3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ Vector3 operator*(const Vector3& a, const float b)
{
	return MakeVector3(b * a.x, b * a.y, b * a.z);
}

__host__ __device__ Vector3 operator*(const float a, const Vector3& b)
{
	return MakeVector3(a * b.x, a * b.y, a * b.z);
}

__host__ __device__ Vector3 operator+(Vector3 a, const Vector3& b)
{
	a += b;
	return a;
}

__host__ __device__ Vector3 operator-(const Vector3& a, const Vector3& b)
{
	return MakeVector3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ Vector3 operator+(const Vector3& a, const float b)
{
	return MakeVector3(a.x + b, a.y + b, a.z + b);
}

__host__ __device__ Vector3 operator-(const Vector3& a, const float b)
{
	return MakeVector3(a.x - b, a.y - b, a.z - b);
}

__host__ __device__ Vector3 operator+(const float a, const Vector3& b)
{
	return MakeVector3(a + b.x, a + b.y, a + b.z);
}

__host__ __device__ Vector3 Normalize(const Vector3& a)
{
	return rsqrt(Dot(a, a)) * a;
}

__host__ __device__ float Clamp(float x, float a, float b)
{
	return min(max(x, a), b);
}

class Matrix44Affine
{
public:
	enum { NumberOfRows = 4 };
	enum { NumberOfCols = 3 };
	enum { NumberOfElement = 12 };

	void Set(float m0, float m1, float m2, float m3, float m4, float m5,
			 float m6, float m7, float m8, float m9, float m10, float m11)
	{
		xAxis.Set(m0, m1, m2);
		yAxis.Set(m3, m4, m5);
		zAxis.Set(m6, m7, m8);
		wAxis.Set(m9, m10, m11);
	}

	void Set(const Vector3& row0, const Vector3& row1, const Vector3& row2, const Vector3& row3)
	{
		xAxis = row0;
		yAxis = row1;
		zAxis = row2;
		wAxis = row3;
	}

	Vector3 xAxis;
	Vector3 yAxis;
	Vector3 zAxis;
	Vector3 wAxis;
};

Matrix44Affine Matrix44AffineFromScaleTranslation(const Vector3& scale, const Vector3& trans)
{
	const float zero(0.0f);
	Matrix44Affine m;
	m.Set(
		scale.x, zero, zero,
		zero, scale.y, zero,
		zero, zero, scale.z,
		trans.x, trans.y, trans.z
		);
	return m;
}

Matrix44Affine Matrix44AffineFromXRotationAngle(float angle)
{
	float s = sin(angle);
	float c = cos(angle);
	const float zero(0.0f);
	const float one(1.0f);
	Matrix44Affine m;
	m.Set(
		one,    zero,   zero,
		zero,  c,       s,
		zero,  -s,      c,
		zero,  zero,   zero
		);
	return m;
}

Matrix44Affine Matrix44AffineFromYRotationAngle(float angle)
{
	float s = sin(angle);
	float c = cos(angle);
	const float zero(0.0f);
	const float one(1.0f);
	Matrix44Affine m;
	m.Set(
		c,      zero,   -s,
		zero,  one, zero,
		s,      zero,   c,
		zero,  zero,   zero );
	return m;
}

Matrix44Affine Matrix44AffineFromZRotationAngle(float angle)
{
	float s = sin(angle);
	float c = cos(angle);
	const float zero(0.0f);
	const float one(1.0f);
	Matrix44Affine m;
	m.Set(
		c,      s,      zero,
		-s,     c,      zero,
		zero,  zero,   one,
		zero,  zero,   zero);
	return m;
}

float Determinant(const Matrix44Affine& m)
{
	return  
		m.xAxis.x * (
		m.yAxis.y * m.zAxis.z -
		m.yAxis.z * m.zAxis.y 
		) + 
		m.xAxis.y * (
		m.yAxis.z * m.zAxis.x -
		m.yAxis.x * m.zAxis.z
		) + 
		m.xAxis.z * (
		m.yAxis.x * m.zAxis.y -
		m.yAxis.y * m.zAxis.x
		);
}

Matrix44Affine Inverse(const Matrix44Affine& m)
{
	const float determinant = Determinant(m);
	if (determinant == 0.0f) 
	{
		return m;
	}
	else
	{
		float determinantRecip = 1.0f / determinant;
		Matrix44Affine ret;
		ret.Set(
				((m.yAxis.y*(m.zAxis.z)+m.yAxis.z*(-m.zAxis.y))*determinantRecip),
				((m.zAxis.y*(m.xAxis.z)+m.zAxis.z*(-m.xAxis.y))*determinantRecip),
				((m.xAxis.y*m.yAxis.z-m.xAxis.z*m.yAxis.y)*determinantRecip),

				((m.yAxis.z*(m.zAxis.x)+m.yAxis.x*(-m.zAxis.z))*determinantRecip),
				((m.zAxis.z*(m.xAxis.x)+m.zAxis.x*(-m.xAxis.z))*determinantRecip),
				((m.yAxis.x*m.xAxis.z-m.xAxis.x*m.yAxis.z)*determinantRecip),

				((m.yAxis.x*(m.zAxis.y)+m.yAxis.y*(-m.zAxis.x))*determinantRecip),
				((m.zAxis.x*(m.xAxis.y)+m.zAxis.y*(-m.xAxis.x))*determinantRecip),
				((m.xAxis.x*m.yAxis.y-m.yAxis.x*m.xAxis.y)*determinantRecip),

				((m.yAxis.x*(m.wAxis.y*m.zAxis.z-m.zAxis.y*m.wAxis.z)+m.yAxis.y*(m.zAxis.x*m.wAxis.z-m.wAxis.x*m.zAxis.z)+m.yAxis.z*(m.wAxis.x*m.zAxis.y-m.zAxis.x*m.wAxis.y))*determinantRecip),
				((m.zAxis.x*(m.xAxis.z*m.wAxis.y-m.xAxis.y*m.wAxis.z)+m.zAxis.y*(m.xAxis.x*m.wAxis.z-m.wAxis.x*m.xAxis.z)+m.zAxis.z*(m.wAxis.x*m.xAxis.y-m.xAxis.x*m.wAxis.y))*determinantRecip),
				((m.wAxis.x*(m.xAxis.z*m.yAxis.y-m.xAxis.y*m.yAxis.z)+m.wAxis.y*(m.xAxis.x*m.yAxis.z-m.yAxis.x*m.xAxis.z)+m.wAxis.z*(m.yAxis.x*m.xAxis.y-m.xAxis.x*m.yAxis.y))*determinantRecip)
			);
		return ret;
	}
}

__host__ __device__ Vector3 TransformPoint(const Vector3& pt, const Matrix44Affine& matrix)
{
	return MakeVector3(matrix.xAxis.x * pt.x + matrix.yAxis.x * pt.y + matrix.zAxis.x * pt.z + matrix.wAxis.x,
					   matrix.xAxis.y * pt.x + matrix.yAxis.y * pt.y + matrix.zAxis.y * pt.z + matrix.wAxis.y,
					   matrix.xAxis.z * pt.x + matrix.yAxis.z * pt.y + matrix.zAxis.z * pt.z + matrix.wAxis.z);
}

__host__ __device__ Vector3 TransformVector(const Vector3& vec, const Matrix44Affine& matrix)
{
	Vector3 transformedPoint = matrix.xAxis * vec.x;
	transformedPoint += matrix.yAxis * vec.y;
	transformedPoint += matrix.zAxis * vec.z;
	return transformedPoint;
}

Matrix44Affine Mult(const Matrix44Affine& m, const Matrix44Affine& b)
{
	Matrix44Affine ret;
	ret.Set(
		TransformVector(m.xAxis, b),
		TransformVector(m.yAxis, b),
		TransformVector(m.zAxis, b),
		TransformPoint(m.wAxis, b)
		);
	return ret;
}

Matrix44Affine operator*(const Matrix44Affine& a, const Matrix44Affine& b)
{
	return Mult(a, b);
}

class Sphere
{
public:
	__device__ Sphere(const Vector3 &position, const float radius) :m_position(position), m_radius(radius) { }

	__device__ float Intersect(const Vector3& R0, const Vector3& Rd, Vector3& hit) const;

	Vector3 m_position;
	float m_radius;
};

__device__ float Sphere::Intersect(const Vector3& R0, const Vector3& Rd, Vector3& hit) const
{
	// Sphere centered at [Xc, Yc, Zc] with radius r satisfies equation [X-Xc]^2 + [Y-Yc]^2 + [Z-Zc]^2 - r^2 = 0
	// Parametric formulation of ray R has equation R(t) = R0 + t*Rd where R0 is an initial position
	// R0 = [X0, Y0, Z0] and Rd is a direction vector Rd = [Xd, Yd, Zd].
	//
	// Letting Rp = [X0-Xc, Y0-Yc, Z0-Zc] we get the solution
	// t^2(Rd dot Rd) + t(2 * Rd dot Rp) + Rp dot Rp - r^2 = 0
	// which can be solved easily through t = (-b +/- (b^2 - 4*a*c)) / 2*a and can discard if the discriminant is < 0
	
	const Vector3 Rp(R0 - m_position);
	const float radiusSqd(m_radius * m_radius);
	const float A(Dot(Rd,Rd));
	const float B(Dot(Rd,Rp) * 2.0f);
	const float C(Dot(Rp,Rp) - radiusSqd);

	const float discrim((B * B) - (A * C * 4.0f));
	
	if (discrim < 0.0f)
	{
		return FLT_MAX;
	}
	else
	{
		const float sqrtDiscrim = sqrt(discrim);
		const float denominator = 1.0f / (2.0f * A);
		const float t1 = (-B + sqrtDiscrim) * denominator;
		const float t2 = (-B - sqrtDiscrim) * denominator;
		const float tMin = min(t1, t2);
		const float tMax = max(t1, t2);
		
		// Cases:
		// a) tMin, tMax < 0, -> t = FLT_MAX
		// b) tMin < 0 && tMax >= 0 -> t = t2
		// c) tMax >= tMin > 0, -> t = t1
		float t = tMax < 0.0f ? FLT_MAX : (tMin < 0.0f ? tMax : tMin);

		hit = Vector3(R0 + (Rd * t));

		return t;
	}
}

// This bound was determined empirically. (i.e. it looks fine with this bound)
__device__ const float boundingSphereRadius = 8.75f;

class SphereFlake
{
public:
	__device__ SphereFlake(unsigned int levelsOfRecursion, const Vector3 &colorA, const Vector3 &colorB)
		: m_level(levelsOfRecursion),
		  m_centralSphere(MakeVector3(0.0f, 0.0f, 0.0f), 5.0f),
		  m_boundingSphere(MakeVector3(0.0f, 0.0f, 0.0f), boundingSphereRadius)
	{
	}

	__device__ float IntersectImpl(const Vector3& R0, const Vector3& Rd, float bound, Vector3& sphereCenter);
	__device__ float Intersect(const Vector3& R0, const Vector3& Rd, Vector3& hit, Vector3& hitNormal, Vector3& hitColor);

	unsigned int m_level;
	Sphere m_centralSphere;
	Sphere m_boundingSphere;

	const static int NUM_CHILDREN = 7;
	static const float scaleFactor;	// Controls how much smaller each child sphere flake is.
	static const float topChildrenZRotation;
	static const float bottomChildrenZRotation;
	static Matrix44Affine childTransforms[NUM_CHILDREN];
	static Matrix44Affine inverseChildTransforms[NUM_CHILDREN];
};

const float SphereFlake::scaleFactor = 1.0f/3.0f;
const float SphereFlake::topChildrenZRotation = 55.0f;
const float SphereFlake::bottomChildrenZRotation = 110.0f;

Matrix44Affine SphereFlake::childTransforms[SphereFlake::NUM_CHILDREN] =
{
	// The first 3 entries are for the "top" children
	Matrix44Affine(
		Matrix44AffineFromScaleTranslation(MakeVector3(scaleFactor, scaleFactor, scaleFactor), MakeVector3(0.0f, 5.0f+5.0f*scaleFactor, 0.0f)) *
		Matrix44AffineFromZRotationAngle( DegToRad(topChildrenZRotation) ) *
		Matrix44AffineFromYRotationAngle( DegToRad(-30.0f-120.0f*0.0f) )
		),

	Matrix44Affine(
		Matrix44AffineFromScaleTranslation(MakeVector3(scaleFactor, scaleFactor, scaleFactor), MakeVector3(0.0f, 5.0f+5.0f*scaleFactor, 0.0f)) *
		Matrix44AffineFromZRotationAngle( DegToRad(topChildrenZRotation) ) *
		Matrix44AffineFromYRotationAngle( DegToRad(-30.0f-120.0f*1.0f) )
		),

	Matrix44Affine(
		Matrix44AffineFromScaleTranslation(MakeVector3(scaleFactor, scaleFactor, scaleFactor), MakeVector3(0.0f, 5.0f+5.0f*scaleFactor, 0.0f)) *
		Matrix44AffineFromZRotationAngle( DegToRad(topChildrenZRotation) ) *
		Matrix44AffineFromYRotationAngle( DegToRad(-30.0f-120.0f*2.0f) )
		),

	// The last 4 entries are for the "bottom" children
	Matrix44Affine(
		Matrix44AffineFromScaleTranslation(MakeVector3(scaleFactor, scaleFactor, scaleFactor), MakeVector3(0.0f, 5.0f+5.0f*scaleFactor, 0.0f)) *
		Matrix44AffineFromZRotationAngle( DegToRad(bottomChildrenZRotation) ) *
		Matrix44AffineFromYRotationAngle( DegToRad(-45.0f-90.0f*0.0f) )
		),

	Matrix44Affine(
		Matrix44AffineFromScaleTranslation(MakeVector3(scaleFactor, scaleFactor, scaleFactor), MakeVector3(0.0f, 5.0f+5.0f*scaleFactor, 0.0f)) *
		Matrix44AffineFromZRotationAngle( DegToRad(bottomChildrenZRotation) ) *
		Matrix44AffineFromYRotationAngle( DegToRad(-45.0f-90.0f*1.0f) )
		),

	Matrix44Affine(
		Matrix44AffineFromScaleTranslation(MakeVector3(scaleFactor, scaleFactor, scaleFactor), MakeVector3(0.0f, 5.0f+5.0f*scaleFactor, 0.0f)) *
		Matrix44AffineFromZRotationAngle( DegToRad(bottomChildrenZRotation) ) *
		Matrix44AffineFromYRotationAngle( DegToRad(-45.0f-90.0f*2.0f) )
		),

	Matrix44Affine(
		Matrix44AffineFromScaleTranslation(MakeVector3(scaleFactor, scaleFactor, scaleFactor), MakeVector3(0.0f, 5.0f+5.0f*scaleFactor, 0.0f)) *
		Matrix44AffineFromZRotationAngle( DegToRad(bottomChildrenZRotation) ) *
		Matrix44AffineFromYRotationAngle( DegToRad(-45.0f-90.0f*3.0f) )
		),

};

Matrix44Affine SphereFlake::inverseChildTransforms[SphereFlake::NUM_CHILDREN] =
{
	Inverse(childTransforms[0]),
	Inverse(childTransforms[1]),
	Inverse(childTransforms[2]),
	Inverse(childTransforms[3]),
	Inverse(childTransforms[4]),
	Inverse(childTransforms[5]),
	Inverse(childTransforms[6]),
};

__constant__ Matrix44Affine deviceChildTransforms[SphereFlake::NUM_CHILDREN];
__constant__ Matrix44Affine deviceInverseChildTransforms[SphereFlake::NUM_CHILDREN];

class ChildSphereStack
{
public:
	static const int StackSize = 100;

	__device__ ChildSphereStack() : m_top(0) {}

	struct ChildSphereInfo
	{
		int level;
		Vector3 localR0;
		Vector3 localRd;
		Vector3 sphereCenter;
	};

	__device__ void Push(int level, const Vector3& newR0, const Vector3& newRd, const Vector3& newSphereCenter)
	{
		m_stack[m_top].level = level;
		m_stack[m_top].localR0 = newR0;
		m_stack[m_top].localRd = newRd;
		m_stack[m_top].sphereCenter = newSphereCenter;
		++m_top;
	};

	__device__ ChildSphereInfo Pop()
	{
		return m_stack[--m_top];
	}

	__device__ bool Empty()
	{
		return m_top == 0;
	}

	__device__ bool Full()
	{
		return m_top >= StackSize;
	}

	int m_top;
	ChildSphereInfo m_stack[StackSize];
};

__device__ float SphereFlake::IntersectImpl(const Vector3& R0, const Vector3& Rd, const float bound, Vector3& sphereCenter)
{
	float result = bound;
	Vector3 hit(MakeVector3(0.0f, 0.0f, 0.0f));

	ChildSphereStack childSphereStack;

	// Now test all the children.
	childSphereStack.Push(0, R0, Rd, MakeVector3(0.0f, 0.0f, 0.0f));
	
	while (!childSphereStack.Empty())
	{
		ChildSphereStack::ChildSphereInfo csi = childSphereStack.Pop();

		const float boundT = m_boundingSphere.Intersect(csi.localR0, csi.localRd, hit);
		if (boundT > result)
		{
			continue;
		}

		const float t = m_centralSphere.Intersect(csi.localR0, csi.localRd, hit);

		if (t < result)
		{
			result = t;
			sphereCenter = csi.sphereCenter;
		}

		if (csi.level >= m_level)
		{
			continue;
		}

		for (int childIdx = 0; childIdx != NUM_CHILDREN; ++childIdx)
		{
			Matrix44Affine& childTransform = deviceChildTransforms[childIdx];
			Matrix44Affine& inverseChildTransform = deviceInverseChildTransforms[childIdx];

			const Vector3 newR0 = TransformPoint(csi.localR0, inverseChildTransform);
			const Vector3 newRd = TransformVector(csi.localRd, inverseChildTransform);
			const Vector3 newSphereCenter = TransformPoint(csi.sphereCenter, childTransform);

			if (!childSphereStack.Full())
			{
				childSphereStack.Push(csi.level + 1, newR0, newRd, newSphereCenter);
			}
		}
	}

	return result;
}

__device__ float SphereFlake::Intersect(const Vector3& R0, const Vector3& Rd, Vector3& hit, Vector3& hitNormal, Vector3& hitColor)
{
	Vector3 sphereCenter(MakeVector3(0.0f, 0.0f, 0.0f));
	float t = IntersectImpl(R0, Rd, FLT_MAX, sphereCenter);

	hit = Rd * t + R0;

	// Calculate the normal
	hitNormal = Normalize(hit - sphereCenter);
	hitColor = MakeVector3(1.0f, 0.2f, 0.1f);

	return t;
}

__device__ Vector3 ShadeRay(const Vector3 &rayStartPos, const Vector3 &rayDirection, int maxLevelsToRecurse=4)
{
	const Vector3 gLightPos(MakeVector3(100.0f, 100.0f, -100.0f));
	const Vector3 gBackgroundColor(MakeVector3(0.0f, 0.0f, 0.2f));
	const float gAmbientIntensity(0.2f);

	SphereFlake sphereFlake(maxLevelsToRecurse, MakeVector3(1.0f,0.3f, 0.3f), MakeVector3(0.0f, 1.0f, 0.0f));
	Vector3 hitVector(MakeVector3(0.0f, 0.0f, 0.0f));
	Vector3 hitNormal(MakeVector3(0.0f, 0.0f, 1.0f));
	Vector3 hitColor(MakeVector3(0.0f, 0.0f, 0.0f));
	if (sphereFlake.Intersect(rayStartPos, rayDirection, hitVector, hitNormal, hitColor) < FLT_MAX)
	{
			const Vector3 closestHit = hitVector;
			const Vector3 closestHitNormal = hitNormal;
			
			// Calculate light direction
			const Vector3 lightDirection(Normalize(gLightPos - closestHit));

			// Calculate light intensity.
			const float lightIntensity = Clamp(Dot(closestHitNormal, lightDirection), gAmbientIntensity, 1.0f);
			
			// Calculate surface color
			return lightIntensity * hitColor;
	}
	else
	{
		return gBackgroundColor;
	}
}

__device__ Vector3 Raytrace(int x, int y)
{
	const float viewWidth(15.0f);
	const float viewHeight(15.0f);
	const float aspectRatio(4.0f/3.0f);
	float xDelta(aspectRatio * viewWidth / width);
	float yDelta(-viewHeight / height);
	Vector3 rayStartPos(MakeVector3(-viewWidth * 0.5f * aspectRatio + float(x) * xDelta, viewHeight * 0.5f + float(y) * yDelta, -20.0f));
	Vector3 rayDirection(MakeVector3(0.0f, 0.0f, 1.0f));

	return ShadeRay(rayStartPos, rayDirection); 
}

__global__ void sphereflake(uint32_t* devPtr, size_t pitch, size_t width, size_t height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	Vector3 pixColor = Raytrace(x, y);
	uint32_t* pix = devPtr + (pitch / sizeof(uint32_t)) * y + x;
	*pix = ((int(pixColor.x * 255.f) & 0xff) << 16) | ((int(pixColor.y * 255.f) & 0xff) << 8) | ((int(pixColor.z * 255.f) & 0xff) << 0);
}

bool outputPPM(const char *fileName, uint32_t *frameContents)
{
	char PPMHeader[256];
	sprintf(PPMHeader, "P3\n" "%d %d\n" "255\n", width, height);
	char *tempBuffer = new char[width*4*3 + 128];

	printf("Writing PPM file to %s...\n", fileName);
	FILE* file = fopen(fileName, "wb");
	if (!file)
	{
		printf("Error opening file!\n");
		return false;
	}

	fwrite(PPMHeader, strlen(PPMHeader), 1, file);
	
	for (unsigned int y = 0; y < height; ++y)
	{
		char *curPos = tempBuffer;
		for (unsigned int x = 0; x < width; ++x)
		{
			unsigned int col = frameContents[y*width + x];
			unsigned int r = col & 0xff;
			unsigned int g = (col >> 8) & 0xff;
			unsigned int b = (col >> 16) & 0xff;
			curPos += sprintf(curPos, "%d %d %d ", r, g, b);
		}

		sprintf(curPos, "\n");
		fwrite(tempBuffer, strlen(tempBuffer), 1, file);
	}
		   
	fclose(file);
	delete [] tempBuffer;
	return true;
}

#define VERIFY_CUDA_SUCCESS(x) do { cudaError result = x; assert(cudaSuccess == result); } while (false);

int main(int argc, char **argv)
{
	size_t pitch = 0;
	uint32_t* devPtr = nullptr;
	VERIFY_CUDA_SUCCESS(cudaMallocPitch(&devPtr, &pitch, width * sizeof(uint32_t), height));

	VERIFY_CUDA_SUCCESS(cudaMemcpyToSymbol(deviceChildTransforms, &SphereFlake::childTransforms, sizeof(deviceChildTransforms)));
	VERIFY_CUDA_SUCCESS(cudaMemcpyToSymbol(deviceInverseChildTransforms, &SphereFlake::inverseChildTransforms, sizeof(deviceInverseChildTransforms)));

	dim3 dimGrid(width / 16, height / 16);
	dim3 dimBlock(16, 16);
	sphereflake<<<dimGrid, dimBlock>>>(devPtr, pitch, width, height);
	
	VERIFY_CUDA_SUCCESS(cudaDeviceSynchronize());

	unique_ptr<uint32_t> output(new uint32_t[width * height]);
	VERIFY_CUDA_SUCCESS(cudaMemcpy2D(output.get(), width * sizeof(uint32_t), devPtr, pitch, width * sizeof(uint32_t), height, cudaMemcpyDeviceToHost));
	outputPPM(imageName, output.get());

	VERIFY_CUDA_SUCCESS(cudaDeviceReset());

	return 0;
}

