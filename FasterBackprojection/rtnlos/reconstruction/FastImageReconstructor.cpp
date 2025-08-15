#include "stdafx.h"
#include "FastImageReconstructor.h"

namespace rtnlos
{
    template <int NROWS, int NCOLS, int NFREQ>
    void FastImageReconstructor<NROWS, NCOLS, NFREQ>::DoWork()
    {
		_workerThread = std::jthread(&FastImageReconstructor<NROWS, NCOLS, NFREQ>::Work, this);
    }

    template<int NROWS, int NCOLS, int NFREQ>
    void FastImageReconstructor<NROWS, NCOLS, NFREQ>::Stop() const
    {
        _incomingFrames.Abort();
    }

    template<int NROWS, int NCOLS, int NFREQ>
    void FastImageReconstructor<NROWS, NCOLS, NFREQ>::Initialize(const SceneParameters& sceneParameters, ViewportSurface* viewportSurface)
    {
        _reconstructor.initialize(DatasetInfo{
            "hidden_scene",
            sceneParameters._apertureFullSize[0],
            sceneParameters._apertureFullSize[1],
            sceneParameters._depthMin,
            sceneParameters._depthMax,
            sceneParameters._depthDelta,
            sceneParameters._depthOffset
        });

        _reconstructor.enableDepthDependentAveraging(_enableDDA);
        _reconstructor.setBandpassInterval(_bandpassInterval.x, _bandpassInterval.y);
        _reconstructor.setNumFrequencies(sceneParameters._numComponents);
        _reconstructor.setWeights(sceneParameters._weights.data());
        _reconstructor.setLambdas(sceneParameters._lambdas.data());
        _reconstructor.setOmegas(sceneParameters._omegas.data());
        _reconstructor.setSamplingSpace(sceneParameters._samplingSpacing);
        _reconstructor.setApertureFullSize(sceneParameters._apertureFullSize.data());
        _reconstructor.setImageDimensions(NROWS, NCOLS);

		_viewportSurface = viewportSurface;
    }

    // This worker should receive Fourier domain histogram for one frame from the FrameHistogramDataQueue
    // It should process that FDH into a 2D reconstructed image using the FastRSD algorithm
    // Once a 2D image is reconstructed, it should be
    // should be packaged and pushed to the outgoing queue.
    template<int NROWS, int NCOLS, int NFREQ>
    void FastImageReconstructor<NROWS, NCOLS, NFREQ>::Work() {

        cudaSetDevice(0);
        cudaFree(nullptr);

        _reconstructor.precalculate();

        FrameHistogramDataPtr histData;
        bool first = true;

        while (!_stop) 
        {
            // Get the next frame of histogram data
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
            _reconstructor.setFFTData(histData->_histogram);
            _reconstructor.reconstructImage(_viewportSurface);
        }

		spdlog::warn("FastImageReconstructor worker thread stopped");
    }

    // Explicit instantiation
    template class FastImageReconstructor<NUMBER_OF_SPAD_ROWS, NUMBER_OF_SPAD_COLS, NUMBER_OF_SPAD_FREQUENCIES>;
}
