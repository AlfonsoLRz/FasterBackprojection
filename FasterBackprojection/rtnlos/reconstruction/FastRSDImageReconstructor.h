#pragma once

#include "data/SceneParameters.h"
#include "data/SensorData.h"
#include "NlosDataProcessor.h"
#include "rsd_reconstructor.h"

namespace rtnlos
{
    // Stage 3: process the Fourier domain histogram into a reconstructed 2D Image using FastRSD.
    template<int NROWS, int NCOLS, int NFREQ>
	class FastRSDImageReconstructor : public NlosDataProcessor
	{
    private:
        bool                            _disableDDA;       // If true, do NOT use depth-dependent averaging.
        FrameHistogramDataQueue&        _incomingFrames;
        ReconstructedImageDataQueue&    _outgoingImages;

        RSDReconstructor                _reconstructor;

    public:
        FastRSDImageReconstructor(FrameHistogramDataQueue& incoming, ReconstructedImageDataQueue& outgoing)
            : _disableDDA(false)
            , _incomingFrames(incoming)
            , _outgoingImages(outgoing)
        {}

        void EnableDepthDependentAveraging(bool enable);
        void DoWork();
        void Initialize(const SceneParameters& sceneParameters);
        void Stop() const;

    protected:
        void Work();
    };
}
