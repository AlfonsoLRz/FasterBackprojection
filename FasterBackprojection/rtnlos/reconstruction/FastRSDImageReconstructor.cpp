#include "stdafx.h"
#include "FastRSDImageReconstructor.h"

namespace NLOS {


    template<int NROWS, int NCOLS, int NFREQ>
    void FastRSDImageReconstructor<NROWS, NCOLS, NFREQ>::InitCmdLineOptions(cxxopts::Options& options)
    {
        options.add_options(m_name)
            ("no-dda", "do NOT use depth dependent averaging", cxxopts::value<bool>(m_disableDDA));
    }

    template<int NROWS, int NCOLS, int NFREQ>
    void FastRSDImageReconstructor<NROWS, NCOLS, NFREQ>::EnableDepthDependentAveraging(bool enable)
    {
        //spdlog::info("Depth Dependent Averaging {}", enable ? "ENABLED" : "DISABLED");
        m_reconstructor.EnableDepthDependentAveraging(enable);
    }

    template<int NROWS, int NCOLS, int NFREQ>
    void FastRSDImageReconstructor<NROWS, NCOLS, NFREQ>::Initialize(const SceneParameters& sceneParameters)
    {
        m_reconstructor.Initialize(DatasetInfo{
            "hidden_scene",
            sceneParameters.ApertureFullSize[0],
            sceneParameters.ApertureFullSize[1],
            sceneParameters.DepthMin,
            sceneParameters.DepthMax,
            sceneParameters.DepthDelta,
            sceneParameters.DepthOffset
        });

        EnableDepthDependentAveraging(!m_disableDDA);
        m_reconstructor.SetNumComponents(sceneParameters.NumComponents);
        m_reconstructor.SetWeights(sceneParameters.Weights.data());
        m_reconstructor.SetLambdas(sceneParameters.Lambdas.data());
        m_reconstructor.SetOmegas(sceneParameters.Omegas.data());
        m_reconstructor.SetIsSimulated(false);
        m_reconstructor.SetSamplingSpace(sceneParameters.SamplingSpacing);
        m_reconstructor.SetAperatureFullsize(sceneParameters.ApertureFullSize.data());

        m_reconstructor.SetImageDimensions(NROWS, NCOLS, NROWS, NCOLS);

        m_reconstructor.PrecalculateRSD();
    }

    // This worker should receive Fourier domain histogram for one frame from the FrameHistogramDataQueue
    // It should process that FDH into a 2D reconstructed image using the FastRSD algorithm
    // Once a 2D image is reconstructed, it should be
    // should be packaged and pushed to the outgoing queue.
    template<int NROWS, int NCOLS, int NFREQ>
    void FastRSDImageReconstructor<NROWS, NCOLS, NFREQ>::Work() {

        m_reconstructor.EnableCubeGeneration(m_logWriter.LogRsdData());

        FrameHistogramDataPtr histData;
        bool first = true;
        while (!m_stopRequested) {
            // get the next frame of histogram data
            if (!m_incomingFrames.Pop(histData)) {
                //spdlog::critical("{:<25}: failed to receive histogram frame data. Exiting.", m_name);
                break;
            }
            if (m_stopRequested) {
                break;
            }

            // reconstruct the FDH into a 2D Image, pushing to the outgoing queue for display
            //spdlog::trace("{:<25}: Received histogram for frame {}, reconstructing image", m_name, histData->FrameNumber);
            auto img = ReconstructedImageDataPtr(new ReconstructedImageDataType(histData->FrameNumber));

            m_reconstructor.SetFFTData(histData->Histogram, NROWS, NCOLS);
            m_reconstructor.ReconstructImage(img->Image2d);
            if (first) { // trash first frame
                first = false;
                continue;
            }

            //spdlog::debug(",{:<25},{},{:8.2f} ms to reconstruct image for frame.", m_name, histData->FrameNumber, timer.Stop());

            // Push 2D Imagein into the outgoing queue
            m_outgoingImages.Push(img);

            // if we are logging rsd cube data, push it into the queue to be logged.
            if (m_logWriter.LogRsdData()) {
                //RsdCubeDataPtr cube(new RsdCubeDataType(histData->FrameNumber, m_reconstructor.GetCubeData()));
                //m_logWriter.PushLog(cube);
            }
        }
        //spdlog::trace("{:<25}: Exiting worker thread", m_name);
    }

    template<int NROWS, int NCOLS, int NFREQ>
    void FastRSDImageReconstructor<NROWS, NCOLS, NFREQ>::OnStop() {
        // there may be a push blocking if the queue is full, so tell the queue to abort it's operation
        //spdlog::trace("{:<25}: Abort", m_name);
        m_incomingFrames.Abort();
        m_outgoingImages.Abort();
    }

    // explicit instantiation
    template class FastRSDImageReconstructor<NUMBER_OF_ROWS, NUMBER_OF_COLS, NUMBER_OF_FREQUENCIES>;
}
