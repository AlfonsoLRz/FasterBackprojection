#include "stdafx.h"
#include "FastRSDImageReconstructor.h"

namespace rtnlos
{
    template<int NROWS, int NCOLS, int NFREQ>
    void FastRSDImageReconstructor<NROWS, NCOLS, NFREQ>::EnableDepthDependentAveraging(bool enable)
    {
        //spdlog::info("Depth Dependent Averaging {}", enable ? "ENABLED" : "DISABLED");
        _reconstructor.EnableDepthDependentAveraging(enable);
    }

    template <int NROWS, int NCOLS, int NFREQ>
    void FastRSDImageReconstructor<NROWS, NCOLS, NFREQ>::DoWork()
    {
		//_workerThread = std::jthread(&FastRSDImageReconstructor<NROWS, NCOLS, NFREQ>::Work, this);
        Work();
    }

    template<int NROWS, int NCOLS, int NFREQ>
    void FastRSDImageReconstructor<NROWS, NCOLS, NFREQ>::Stop() const
    {
        _incomingFrames.Abort();
        _outgoingImages.Abort();
    }

    template<int NROWS, int NCOLS, int NFREQ>
    void FastRSDImageReconstructor<NROWS, NCOLS, NFREQ>::Initialize(const SceneParameters& sceneParameters)
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

        EnableDepthDependentAveraging(!_disableDDA);
        _reconstructor.SetNumComponents(sceneParameters._numComponents);
        _reconstructor.SetWeights(sceneParameters._weights.data());
        _reconstructor.SetLambdas(sceneParameters._lambdas.data());
        _reconstructor.SetOmegas(sceneParameters._omegas.data());
        _reconstructor.SetIsSimulated(false);
        _reconstructor.SetSamplingSpace(sceneParameters._samplingSpacing);
        _reconstructor.SetApertureFullsize(sceneParameters._apertureFullSize.data());
        _reconstructor.SetImageDimensions(NROWS, NCOLS, NROWS, NCOLS);
        _reconstructor.PrecalculateRSD();
    }

    // This worker should receive Fourier domain histogram for one frame from the FrameHistogramDataQueue
    // It should process that FDH into a 2D reconstructed image using the FastRSD algorithm
    // Once a 2D image is reconstructed, it should be
    // should be packaged and pushed to the outgoing queue.
    template<int NROWS, int NCOLS, int NFREQ>
    void FastRSDImageReconstructor<NROWS, NCOLS, NFREQ>::Work() {

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

            // Reconstruct the FDH into a 2D Image, pushing to the outgoing queue for display
            auto img = std::make_shared<ReconstructedImageDataType>(histData->_frameNumber);

            _reconstructor.SetFFTData(histData->_histogram, NROWS, NCOLS);
            _reconstructor.ReconstructImage(img->_image);
            if (first)  // Trash first frame
            { 
                first = false;
                continue;
            }

            // Push a new mage in into the outgoing queue
            //_outgoingImages.Push(img);
        }

		spdlog::warn("FastRSDImageReconstructor worker thread stopped");
    }

    // Explicit instantiation
    template class FastRSDImageReconstructor<NUMBER_OF_ROWS, NUMBER_OF_COLS, NUMBER_OF_FREQUENCIES>;
}
