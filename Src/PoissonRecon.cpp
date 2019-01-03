/*
Copyright (c) 2006, Michael Kazhdan and Matthew Bolitho
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#ifdef _WIN32
#include <Windows.h>
#include <Psapi.h>
#endif // _WIN32
#include "MyTime.h"
#include "MarchingCubes.h"
#include "Octree.h"
#include "SparseMatrix.h"
#include "CmdLineParser.h"
#include "PPolynomial.h"
#include "Ply.h"
#include "MemoryUsage.h"
#ifdef _OPENMP
#include "omp.h"
#endif // _OPENMP

using namespace poisson_recon;

void DumpOutput( const char* format , ... );
void DumpOutput2( std::vector< char* >& comments , const char* format , ... );
#include "MultiGridOctreeData.h"

#define DEFAULT_FULL_DEPTH 5

#define XSTR(x) STR(x)
#define STR(x) #x
#if DEFAULT_FULL_DEPTH
#pragma message ( "[WARNING] Setting default full depth to " XSTR(DEFAULT_FULL_DEPTH) )
#endif // DEFAULT_FULL_DEPTH

#include <stdarg.h>
#include "../include/PoissonRecon.h"
char* outputFile=NULL;
int echoStdout=0;
void DumpOutput( const char* format , ... )
{
	if( outputFile )
	{
		FILE* fp = fopen( outputFile , "a" );
		va_list args;
		va_start( args , format );
		vfprintf( fp , format , args );
		fclose( fp );
		va_end( args );
	}
	if( echoStdout )
	{
		va_list args;
		va_start( args , format );
		vprintf( format , args );
		va_end( args );
	}
}
void DumpOutput2( std::vector< char* >& comments  , const char* format , ... )
{
	if( outputFile )
	{
		FILE* fp = fopen( outputFile , "a" );
		va_list args;
		va_start( args , format );
		vfprintf( fp , format , args );
		fclose( fp );
		va_end( args );
	}
	if( echoStdout )
	{
		va_list args;
		va_start( args , format );
		vprintf( format , args );
		va_end( args );
	}
	comments.push_back( new char[1024] );
	char* str = comments.back();
	va_list args;
	va_start( args , format );
	vsprintf( str , format , args );
	va_end( args );
	if( str[strlen(str)-1]=='\n' ) str[strlen(str)-1] = 0;
}

Point3D< unsigned char > ReadASCIIColor( FILE* fp )
{
	Point3D< unsigned char > c;
	if( fscanf( fp , " %c %c %c " , &c[0] , &c[1] , &c[2] )!=3 ) fprintf( stderr , "[ERROR] Failed to read color\n" ) , exit( 0 );
	return c;
}

PlyProperty PlyColorProperties[]=
{
	{ "r"     , PLY_UCHAR , PLY_UCHAR , int( offsetof( Point3D< unsigned char > , coords[0] ) ) , 0 , 0 , 0 , 0 } ,
	{ "g"     , PLY_UCHAR , PLY_UCHAR , int( offsetof( Point3D< unsigned char > , coords[1] ) ) , 0 , 0 , 0 , 0 } ,
	{ "b"     , PLY_UCHAR , PLY_UCHAR , int( offsetof( Point3D< unsigned char > , coords[2] ) ) , 0 , 0 , 0 , 0 } ,
	{ "red"   , PLY_UCHAR , PLY_UCHAR , int( offsetof( Point3D< unsigned char > , coords[0] ) ) , 0 , 0 , 0 , 0 } , 
	{ "green" , PLY_UCHAR , PLY_UCHAR , int( offsetof( Point3D< unsigned char > , coords[1] ) ) , 0 , 0 , 0 , 0 } ,
	{ "blue"  , PLY_UCHAR , PLY_UCHAR , int( offsetof( Point3D< unsigned char > , coords[2] ) ) , 0 , 0 , 0 , 0 }
};

bool ValidPlyColorProperties( const bool* props ){ return ( props[0] || props[3] ) && ( props[1] || props[4] ) && ( props[2] || props[5] ); }

template< class Real, int Degree, class Vertex >
int _Execute(const PoissonReconParameters & parameters, CoredVectorMeshData< Vertex>& mesh)
{
	// reset the static variables (could unstaticize them)
	Reset< Real >();

	std::vector< char* > comments;

	if(parameters.verbose) echoStdout=1;

	XForm4x4< Real > xForm , iXForm;
	xForm = XForm4x4< Real >::Identity();
	iXForm = xForm.inverse();

	DumpOutput2( comments , "Running Screened Poisson Reconstruction (Version 8.0)\n" );
	
	double t;
	double tt=Time();
	Real isoValue = 0;

	Octree< Real > tree;
	tree.threads = parameters.threads;

	int maxSolveDepth{ parameters.depth };
	if (parameters.maxSolveDepth.set())
		maxSolveDepth = *parameters.maxSolveDepth;
	
	OctNode< TreeNodeData >::SetAllocator( MEMORY_ALLOCATOR_BLOCK_SIZE );

	t=Time();
	int kernelDepth = parameters.kernelDepth.set() ?  *parameters.kernelDepth : parameters.depth-2;
	if( kernelDepth>parameters.depth)
	{
		fprintf( stderr,"[ERROR] kernelDepth can't be greater than depth: %d <= %d\n" , *parameters.kernelDepth, parameters.depth );
		kernelDepth = parameters.depth;
	}

	double maxMemoryUsage;
	t=Time() , tree.maxMemoryUsage=0;
	SparseNodeData< PointData< Real > , 0 >* pointInfo = new SparseNodeData< PointData< Real > , 0 >();
	SparseNodeData< Point3D< Real > , NORMAL_DEGREE >* normalInfo = new SparseNodeData< Point3D< Real > , NORMAL_DEGREE >();
	SparseNodeData< Real , WEIGHT_DEGREE >* densityWeights = new SparseNodeData< Real , WEIGHT_DEGREE >();
	SparseNodeData< Real , NORMAL_DEGREE >* nodeWeights = new SparseNodeData< Real , NORMAL_DEGREE >();
	int pointCount;
	typedef typename Octree< Real >::template ProjectiveData< Point3D< Real > > ProjectiveColor;
	SparseNodeData< ProjectiveColor , DATA_DEGREE >* colorData = NULL;

	if( parameters.color.set() && *parameters.color > 0 )
	{
		colorData = new SparseNodeData< ProjectiveColor , DATA_DEGREE >();
		pointCount = tree.template SetTree< float, NORMAL_DEGREE , WEIGHT_DEGREE , DATA_DEGREE , Point3D< unsigned char >>( 
			parameters.pointStreamWithData , parameters.minDepth, parameters.depth , parameters.fullDepth, kernelDepth , Real(parameters.samplesPerNode) , parameters.scale , parameters.confidence , parameters.normalWeights , parameters.pointWeight, parameters.adaptiveExponent, *densityWeights , *pointInfo , *normalInfo , *nodeWeights , colorData , xForm , parameters.dirichlet , parameters.complete);

		for( const OctNode< TreeNodeData >* n = tree.tree().nextNode() ; n!=NULL ; n=tree.tree().nextNode( n ) )
		{
			int idx = colorData->index( n );
			if( idx>=0 ) colorData->data[idx] *= (Real)pow( *parameters.color , n->depth() );
		}
	}
	else
	{
		pointCount = tree.template SetTree< float, NORMAL_DEGREE , WEIGHT_DEGREE , DATA_DEGREE , Point3D< unsigned char >>( 
			parameters.pointStream , parameters.minDepth , parameters.depth , parameters.fullDepth , kernelDepth , Real(parameters.samplesPerNode) , parameters.scale , parameters.confidence , parameters.normalWeights , parameters.pointWeight , parameters.adaptiveExponent , *densityWeights , *pointInfo , *normalInfo , *nodeWeights , colorData, xForm , parameters.dirichlet, parameters.complete );
	}
	if( !parameters.density ) 
		delete densityWeights , densityWeights = NULL;
	{
		std::vector< int > indexMap;
		if( NORMAL_DEGREE>Degree ) tree.template EnableMultigrid< NORMAL_DEGREE >( &indexMap );
		else                       tree.template EnableMultigrid<        Degree >( &indexMap );
		if( pointInfo ) pointInfo->remapIndices( indexMap );
		if( normalInfo ) normalInfo->remapIndices( indexMap );
		if( densityWeights ) densityWeights->remapIndices( indexMap );
		if( nodeWeights ) nodeWeights->remapIndices( indexMap );
		if( colorData ) colorData->remapIndices( indexMap );
	}

	DumpOutput2( comments , "#             Tree set in: %9.1f (s), %9.1f (MB)\n" , Time()-t , tree.maxMemoryUsage );
	DumpOutput( "Input Points: %d\n" , pointCount );
	DumpOutput( "Leaves/Nodes: %d/%d\n" , tree.leaves() , tree.nodes() );
	DumpOutput( "Memory Usage: %.3f MB\n" , float( MemoryInfo::Usage() )/(1<<20) );

	maxMemoryUsage = tree.maxMemoryUsage;
	t=Time() , tree.maxMemoryUsage=0;
	DenseNodeData< Real , Degree > constraints = tree.template SetLaplacianConstraints< Degree >( *normalInfo );
	delete normalInfo;
	DumpOutput2( comments , "#      Constraints set in: %9.1f (s), %9.1f (MB)\n" , Time()-t , tree.maxMemoryUsage );
	DumpOutput( "Memory Usage: %.3f MB\n" , float( MemoryInfo::Usage())/(1<<20) );
	maxMemoryUsage = std::max< double >( maxMemoryUsage , tree.maxMemoryUsage );

	t=Time() , tree.maxMemoryUsage=0;
	DenseNodeData< Real , Degree > solution = tree.SolveSystem( *pointInfo , constraints , parameters.showResidual , parameters.iters , maxSolveDepth , parameters.cgDepth, parameters.csSolverAccuracy );
	delete pointInfo;
	constraints.resize( 0 );

	DumpOutput2( comments , "# Linear system solved in: %9.1f (s), %9.1f (MB)\n" , Time()-t , tree.maxMemoryUsage );
	DumpOutput( "Memory Usage: %.3f MB\n" , float( MemoryInfo::Usage() )/(1<<20) );
	maxMemoryUsage = std::max< double >( maxMemoryUsage , tree.maxMemoryUsage );

	if( parameters.verbose ) tree.maxMemoryUsage=0;
	t=Time();
	isoValue = tree.GetIsoValue( solution , *nodeWeights );
	delete nodeWeights;
	DumpOutput( "Got average in: %f\n" , Time()-t );
	DumpOutput( "Iso-Value: %e\n" , isoValue );

	if( parameters.voxel_grid_file.set() )
	{
		double t = Time();
		FILE* fp = fopen(parameters.voxel_grid_file->c_str(), "wb" );
		if( !fp ) fprintf( stderr , "Failed to open voxel file for writing: %s\n" , parameters.voxel_grid_file->c_str());
		else
		{
			int res = 0;
			Pointer( Real ) values = tree.Evaluate( solution , res , isoValue , parameters.voxelDepth , parameters.primalVoxel );
			fwrite( &res , sizeof(int) , 1 , fp );
			if( sizeof(Real)==sizeof(float) ) fwrite( values , sizeof(float) , res*res*res , fp );
			else
			{
				float *fValues = new float[res*res*res];
				for( int i=0 ; i<res*res*res ; i++ ) fValues[i] = float( values[i] );
				fwrite( fValues , sizeof(float) , res*res*res , fp );
				delete[] fValues;
			}
			fclose( fp );
			DeletePointer( values );
		}
		DumpOutput( "Got voxel grid in: %f\n" , Time()-t );
	}
	
	t = Time() , tree.maxMemoryUsage = 0;
	tree.template GetMCIsoSurface< Degree , WEIGHT_DEGREE , DATA_DEGREE >( densityWeights, colorData, solution , isoValue , mesh , !parameters.linearFit , !parameters.nonManifold , parameters.polygonMesh );
	if( parameters.polygonMesh ) DumpOutput2( comments , "#         Got polygons in: %9.1f (s), %9.1f (MB)\n" , Time()-t , tree.maxMemoryUsage );
	else                  DumpOutput2( comments , "#        Got triangles in: %9.1f (s), %9.1f (MB)\n" , Time()-t , tree.maxMemoryUsage );
	maxMemoryUsage = std::max< double >( maxMemoryUsage , tree.maxMemoryUsage );
	DumpOutput2( comments , "#             Total Solve: %9.1f (s), %9.1f (MB)\n" , Time()-tt , maxMemoryUsage );

	solution.resize( 0 );
	if( colorData ){ delete colorData ; colorData = NULL; }
	return 1;
}

template< class Real, class Vertex >
int Execute(const PoissonReconParameters & parameters, CoredVectorMeshData< Vertex>& mesh)
{
	switch( parameters.splineDegree )
	{
	case 1: return _Execute< Real , 1 , Vertex >( parameters , mesh );
	case 2: return _Execute< Real , 2 , Vertex >( parameters , mesh );
	case 3: return _Execute< Real , 3 , Vertex >( parameters , mesh );
	case 4: return _Execute< Real , 4 , Vertex >( parameters , mesh );
	default:
		fprintf( stderr , "[ERROR] Only B-Splines of degree 1 - 4 are supported" );
		return EXIT_FAILURE;
	}
}

#ifdef _WIN32
inline double to_seconds( const FILETIME& ft )
{
	const double low_to_sec=100e-9; // 100 nanoseconds
	const double high_to_sec=low_to_sec*4294967296.0;
	return ft.dwLowDateTime*low_to_sec+ft.dwHighDateTime*high_to_sec;
}
#endif // _WIN32

int poisson_recon::performReconstruction(const PoissonReconParameters & parameters, CoredVectorMeshData<poisson_recon::PlyColorVertex<float>>& mesh)
{
	Execute<float, PlyColorVertex<float>>(parameters, mesh);

	return EXIT_SUCCESS;
}

int poisson_recon::getDefaultThreadsCount()
{
	return omp_get_num_procs();
}
