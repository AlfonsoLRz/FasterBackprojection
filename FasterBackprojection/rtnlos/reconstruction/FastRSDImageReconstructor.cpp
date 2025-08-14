#include "stdafx.h"
#include "FastRSDImageReconstructor.h"

#include "../../ViewportSurface.h"

namespace rtnlos
{
    template <int NROWS, int NCOLS, int NFREQ>
    void FastRSDImageReconstructor<NROWS, NCOLS, NFREQ>::DoWork()
    {
		_workerThread = std::jthread(&FastRSDImageReconstructor<NROWS, NCOLS, NFREQ>::Work, this);
    }

    template<int NROWS, int NCOLS, int NFREQ>
    void FastRSDImageReconstructor<NROWS, NCOLS, NFREQ>::Stop() const
    {
        _incomingFrames.Abort();
    }

    template<int NROWS, int NCOLS, int NFREQ>
    void FastRSDImageReconstructor<NROWS, NCOLS, NFREQ>::Initialize(const SceneParameters& sceneParameters, ViewportSurface* viewportSurface)
    {
        _reconstructor.Initialize(DatasetInfo{
            "hidden_scene",
            sceneParameters._apertureFullSize[0],
            sceneParameters._apertureFullSize[1],
            sceneParameters._depthMin,
            sceneParameters._depthMax,
            sceneParameters._depthDelta,
            sceneParameters._depthOffset
        });

        _reconstructor.EnableDepthDependentAveraging(_enableDDA);
        _reconstructor.SetBandpassInterval(_bandpassInterval.x, _bandpassInterval.y);
        _reconstructor.SetNumFrequencies(sceneParameters._numComponents);
        _reconstructor.SetWeights(sceneParameters._weights.data());
        _reconstructor.SetLambdas(sceneParameters._lambdas.data());
        _reconstructor.SetOmegas(sceneParameters._omegas.data());
        _reconstructor.SetSamplingSpace(sceneParameters._samplingSpacing);
        _reconstructor.SetApertureFullSize(sceneParameters._apertureFullSize.data());
        _reconstructor.SetImageDimensions(NROWS, NCOLS);

		_viewportSurface = viewportSurface;
    }

    // This worker should receive Fourier domain histogram for one frame from the FrameHistogramDataQueue
    // It should process that FDH into a 2D reconstructed image using the FastRSD algorithm
    // Once a 2D image is reconstructed, it should be
    // should be packaged and pushed to the outgoing queue.
    template<int NROWS, int NCOLS, int NFREQ>
    void FastRSDImageReconstructor<NROWS, NCOLS, NFREQ>::Work() {

        cudaSetDevice(0); // or whatever device you're using
        cudaFree(nullptr);

        _reconstructor.PrecalculateRSD();
        _reconstructor.EnableCubeGeneration(false);

        FrameHistogramDataPtr histData;
        bool first = true;

        while (!_stop) 
        {
            // get the next frame of histogram data
            if (!_incomingFrames.Pop(histData)) 
                break;
  
            if (_stop)
                break;

            if (first)
            {
	            first = false;
                continue;
            }

            // Reconstruct the FDH into a 2D Image, pushing to the outgoing queue for display
            auto image = std::make_shared<ReconstructedImageDataType>(histData->_frameNumber);
            _reconstructor.SetFFTData(histData->_histogram);
            _reconstructor.ReconstructImage(_viewportSurface);
        }

		spdlog::warn("FastRSDImageReconstructor worker thread stopped");
    }

    // Explicit instantiation
    template class FastRSDImageReconstructor<NUMBER_OF_SPAD_ROWS, NUMBER_OF_SPAD_COLS, NUMBER_OF_SPAD_FREQUENCIES>;
}
