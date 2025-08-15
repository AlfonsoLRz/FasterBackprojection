#pragma once

#include "data/SceneParameters.h"
#include "data/SensorData.h"

#include "FastRSDReconstruction.h"
#include "NlosDataProcessor.h"

class ViewportSurface;

namespace rtnlos
{
    // Stage 3: process the Fourier domain histogram into a reconstructed 2D Image using FastRSD.
    template<int NROWS, int NCOLS, int NFREQ>
	class FastImageReconstructor : public NlosDataProcessor
	{
    private:
        bool                            _enableDDA;        // If true, do NOT use depth-dependent averaging
        glm::vec2                       _bandpassInterval;

        FrameHistogramDataQueue&        _incomingFrames;
        FastRSDReconstruction           _reconstructor;
		ViewportSurface*                _viewportSurface;    // The viewport surface to draw the reconstructed image into

    public:
        explicit FastImageReconstructor(FrameHistogramDataQueue& incoming)
	        : _enableDDA(true), _bandpassInterval(0.1f, 0.9f), _incomingFrames(incoming), _viewportSurface(nullptr)
        {
        }

        void DoWork();
        void Initialize(const SceneParameters& sceneParameters, ViewportSurface* viewportSurface);
        void Stop() const;

    protected:
        void Work();
    };
}
