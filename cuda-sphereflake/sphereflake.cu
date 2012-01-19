#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <memory>
#include <limits>
#include <math.h>

using namespace std;

const unsigned int width = 640;
const unsigned int height = (width * 3) / 4;	// Maintain a 4/3 aspect ratio

const char* imageName = "sphereflake.ppm";

class Vector3
{
public:
	__device__ Vector3(const float x_, const float y_, const float z_) : x(x_), y(y_), z(z_) {}

	float x;
	float y;
	float z;
};

__device__ float Dot(const Vector3& a, const Vector3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ Vector3 operator*(const Vector3& a, const float b)
{
	return Vector3(b * a.x, b * a.y, b * a.z);
}

__device__ Vector3 operator*(const float a, const Vector3& b)
{
	return Vector3(a * b.x, a * b.y, a * b.z);
}

__device__ Vector3 operator+(const Vector3& a, const Vector3& b)
{
	return Vector3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ Vector3 operator-(const Vector3& a, const Vector3& b)
{
	return Vector3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ Vector3 operator+(const Vector3& a, const float b)
{
	return Vector3(a.x + b, a.y + b, a.z + b);
}

__device__ Vector3 operator-(const Vector3& a, const float b)
{
	return Vector3(a.x - b, a.y - b, a.z - b);
}

__device__ Vector3 operator+(const float a, const Vector3& b)
{
	return Vector3(a + b.x, a + b.y, a + b.z);
}

__device__ Vector3 Normalize(const Vector3& a)
{
	return rsqrt(Dot(a, a)) * a;
}

__device__ float Clamp(float x, float a, float b)
{
	return min(max(x, a), b);
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
	
	Vector3 Rp(R0 - m_position);
	float radiusSqd(m_radius * m_radius);
	float A(Dot(Rd,Rd));
	float B(Dot(Rd,Rp) * 2.0f);
	float C(Dot(Rp,Rp) - radiusSqd);

	// gVecFloat_Four is a constant VecFloat with the value 4.0f
	float discrim((B * B) - (A * C * 4.0f));
	
	if (discrim < 0.0f)
	{
		return FLT_MAX;
	}
	else
	{
		float sqrtDiscrim = sqrt(discrim);
		float denominator = 1.0f / (2.0f * A);
		float t1 = (-B + sqrtDiscrim) * denominator;
		float t2 = (-B - sqrtDiscrim) * denominator;
		float t;

		// Cases:
		// a) t1, t2 < 0, -> return -1.0f
		// b) t1 < 0 && t2 > 0 -> t = t2
		// c) t2 < 0 && t1 > 0, -> t = t1
		// d) t2 >= t1 > 0, -> t = t1
		// e) t1 > t2 > 0, -> t = t2
		
		if (t1 <= 0.0f || t2 <= 0.0f)
		{
			if (t1 > 0.0f)
			{
				t = t1;
			}
			else if(t2 > 0.0f)
			{
				t = t2;
			}
			else
			{
				t = FLT_MAX;
			}
		}
		else
		{
			t = min(t1, t2);
		}

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
		  m_centralSphere(Vector3(0.0f, 0.0f, 0.0f), 5.0f),
		  m_boundingSphere(Vector3(0.0f, 0.0f, 0.0f), boundingSphereRadius)
	{
	}

	__device__ float Intersect(const Vector3& R0, const Vector3& Rd, Vector3& hit, Vector3& hitNormal, Vector3& hitColor)
	{
		const float t = m_centralSphere.Intersect(R0, Rd, hit);
		Vector3 sphereCenter(m_centralSphere.m_position);
		hitNormal = Normalize(hit - sphereCenter);
		hitColor = Vector3(1.0f, 0.2f, 0.1f);
		return t;
	}

	unsigned int m_level;
	Sphere m_centralSphere;
	Sphere m_boundingSphere;
};

__device__ Vector3 ShadeRay(const Vector3 &rayStartPos, const Vector3 &rayDirection, int maxLevelsToRecurse=10)
{
	const Vector3 gLightPos(100.0f, 100.0f, -100.0f);
	const Vector3 gBackgroundColor(0.0f, 0.0f, 0.2f);
	const float gAmbientIntensity(0.2f);

	SphereFlake sphereFlake(maxLevelsToRecurse, Vector3(1.0f,0.3f, 0.3f), Vector3(0.0f, 1.0f, 0.0f));
	Vector3 hitVector(0.0f, 0.0f, 0.0f);
	Vector3 hitNormal(0.0f, 0.0f, 1.0f);
	Vector3 hitColor(0.0f, 0.0f, 0.0f);
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
	Vector3 rayStartPos(-viewWidth * 0.5f * aspectRatio + float(x) * xDelta, viewHeight * 0.5f + float(y) * yDelta, -20.0f);
	Vector3 rayDirection(0.0f, 0.0f, 1.0f);

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

int main(int argc, char **argv)
{
	size_t pitch = 0;
	uint32_t* devPtr = nullptr;
	cudaMallocPitch(&devPtr, &pitch, width * sizeof(uint32_t), height);

	dim3 dimGrid(width / 16, height / 16);
	dim3 dimBlock(16, 16);
	sphereflake<<<dimGrid, dimBlock>>>(devPtr, pitch, width, height);
	
	cudaDeviceSynchronize();

	unique_ptr<uint32_t> output(new uint32_t[width * height]);
	cudaMemcpy2D(output.get(), width * sizeof(uint32_t), devPtr, pitch, width * sizeof(uint32_t), height, cudaMemcpyDeviceToHost);
	outputPPM(imageName, output.get());

	cudaDeviceReset();

	return 0;
}

