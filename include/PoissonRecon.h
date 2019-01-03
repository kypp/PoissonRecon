#pragma once
#include <string>
#include <optional>
#include "Geometry.h"
#include "Ply.h"

namespace poisson_recon {

	int getDefaultThreadsCount();

	struct PoissonReconParameters 
	{
		OrientedPointStreamWithData< float, Point3D< unsigned char > >* pointStreamWithData;
		OrientedPointStream< float >* pointStream;


		std::optional<std::string> voxel_grid_file;

		bool double_precision{ false };
		bool performance{ false };
		bool complete{ false };
		bool showResidual{ false };
		bool noComments{ false };
		bool polygonMesh{ false };
		bool confidence{ false };
		bool normalWeights{ false };
		bool nonManifold{ false };
		bool density{ false };
		bool verbose{ false };
		bool ASCII{ false };
		bool dirichlet{ false };
		bool linearFit{ false };
		bool primalVoxel{ false };

		int splineDegree{ 2 };
		int depth{ 8 };
		int cgDepth{ 0 };
		std::optional<int> kernelDepth;
		int adaptiveExponent{ 1 };
		int iters{ 8 };
		int voxelDepth{ -1 };
		int fullDepth{ 5 };
		int minDepth{ 0 };
		std::optional<int> maxSolveDepth;
		int threads{ getDefaultThreadsCount() };

		std::optional<float> color; // 16
		float samplesPerNode{ 1.5f };
		float scale{ 1.1f };
		float csSolverAccuracy{ float(1e-3) };
		float pointWeight{ 4.f };
	};

	int performReconstruction(const PoissonReconParameters & parameters, CoredVectorMeshData<PlyColorVertex<float>>& mesh);
}