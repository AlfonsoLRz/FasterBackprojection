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
    public:
		enum FastReconstructionType : uint8_t
        {
	        RSD,
            LCT,
			NUM_ALGORITHMS
        };

    private:
        inline static FastReconstructionAlgorithm* _reconstructor[FastReconstructionType::NUM_ALGORITHMS] = {
            new FastRSDReconstruction,
			nullptr
		};

        bool                                    _enableDDA;             // If true, do NOT use depth-dependent averaging
        glm::vec2                               _bandpassInterval;

        FrameHistogramDataQueue&                _incomingFrames;
		ViewportSurface*                        _viewportSurface;       // The viewport surface to draw the reconstructed image into

        FastReconstructionType                  _reconstructionAlgorithm;

    public:
        explicit FastImageReconstructor(FrameHistogramDataQueue& incoming);
    	virtual ~FastImageReconstructor();

        void DoWork();
        void Initialize(const SceneParameters& sceneParameters, ViewportSurface* viewportSurface);
        void Stop() const;

        void SetReconstructionAlgorithm(FastReconstructionType algorithm);

    protected:
        void Work();
    };
}
