#pragma once

#include "../types.h"
#include "../util/DataProcessor.h"
#include "../util/LogWriter.h"
#include "../data/RsdCubeData.h"
#include "rsd_reconstructor.h"

namespace NLOS {

    // stage 3: process the Fourier domain histogram into a reconstructed 2D Image using FastRSD.
    template<int NROWS, int NCOLS, int NFREQ>
    class FastRSDImageReconstructor : public DataProcessor {
    public:
        FastRSDImageReconstructor(FrameHistogramDataQueue& incoming, ReconstructedImageDataQueue& outgoing, LogWriter& logWriter)
            : DataProcessor("FastRSDImageReconstructor", logWriter)
            , m_incomingFrames(incoming)
            , m_outgoingImages(outgoing)
            , m_disableDDA(false)
        {}

        void EnableDepthDependentAveraging(bool enable);

    protected:
        virtual void InitCmdLineOptions(cxxopts::Options& options);
        virtual void Initialize(const SceneParameters& sceneParameters);
        virtual void Work();
        virtual void OnStop();

    private:
        bool m_disableDDA; // if true, do NOT use depth-dependent averaging.
        FrameHistogramDataQueue& m_incomingFrames;
        ReconstructedImageDataQueue& m_outgoingImages;

        RSDReconstructor m_reconstructor;
    };
}
