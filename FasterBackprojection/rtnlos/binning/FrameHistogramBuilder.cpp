#include "stdafx.h"
#include "FrameHistogramBuilder.h"

#include "cufft.h"

namespace rtnlos
{
    template<int NINDICES, int NFREQ>
    void FrameHistogramBuilder<NINDICES, NFREQ>::Initialize(const rtnlos::SceneParameters& sceneParameters)
    {
        if (sceneParameters._freMask.size() != NFREQ) 
            throw std::logic_error(
                fmt::format("Incorrect number of frequencies. Try rebuilding exe with NUMBER_OF_FREQUENCIES = {} in compile_time_constants.h",
                sceneParameters._freMask.size()));

        if (sceneParameters._apertureWidth * sceneParameters._apertureHeight != NINDICES)
            throw std::logic_error(fmt::format("Incorrect grid dimension. Try rebuilding exe with NUMBER_OF_ROWS={} and NUMBER_OF_COLS={} in compile_time_constants.h",
                sceneParameters._apertureWidth, sceneParameters._apertureHeight));


        const float downsamplingMultiplier = 1.f / (1 << sceneParameters._downsamplingRate);
        const float ts = sceneParameters._resolution * sceneParameters.PICOSECOND / downsamplingMultiplier;
        const float photonMaxTime = std::round(2 * sceneParameters._depthMax / (ts * sceneParameters.LIGHT_SPEED));
        const float* frequencyMask = sceneParameters._freMask.data();

        for (int i = 0; i < NFREQ; i++) 
            // Incorporate -2*pi/maxT into each frequency to save 2 multiplications on each record.
            _frequencies[i] = frequencyMask[i] * static_cast<float>(-2. * glm::pi<float>() / photonMaxTime);

#if USE_FDH_LOOKUP_TABLE
        _photonMinTime = static_cast<int>(sceneParameters._zGate * downsamplingMultiplier);
        _photonMaxTime = static_cast<int>(photonMaxTime);
        _numTimes = _photonMaxTime - _photonMinTime + 1;

        BuildLookupTables();
#endif
    }

    template <int NINDICES, int NFREQ>
    void FrameHistogramBuilder<NINDICES, NFREQ>::DoWork()
    {
        spdlog::info("Starting FrameHistogramBuilder worker thread");
		_workerThread = std::jthread(&FrameHistogramBuilder<NINDICES, NFREQ>::Work, this);
    }

#if USE_FDH_LOOKUP_TABLE
    template<int NINDICES, int NFREQ>
    void FrameHistogramBuilder<NINDICES, NFREQ>::BuildLookupTables()
	{
        _sinLookup.resize(NFREQ * _numTimes);
        _cosLookup.resize(NFREQ * _numTimes);

        for (int t = 0; t < _numTimes; t++) 
        {
            for (int f = 0; f < NFREQ; f++) 
            {
                _sinLookup[t * NFREQ + f] = std::sin(_frequencies[f] * (t + _photonMinTime));
                _cosLookup[t * NFREQ + f] = std::cos(_frequencies[f] * (t + _photonMinTime));
            }
        }
    }
#endif

    // This worker should receive T3Rec data from the the RawSensorDataQueue
    // It should parse the data and bin the data into a Fourier domain histogram.
    // The incoming data should be assumed to be in random sizes, and not a full frame at a time.
    // Once an entire frame is recieved and binned, the histogram for that frame
    // should be packaged and pushed to the outgoing queue.
    template<int NINDICES, int NFREQ>
    void FrameHistogramBuilder<NINDICES, NFREQ>::Work()
	{
		rtnlos::ParsedSensorDataPtr parsedData;

        while (!_stop) 
        {
            // get the next frame of parsed photon data
            if (!_incomingParsed.Pop(parsedData)) 
            {
                spdlog::critical("failed to receive parsed photon data. Exiting.");
                break;
            }

            spdlog::stopwatch timer;
            spdlog::trace("Received parsed data for frame {}, going to build FDH", parsedData->_frameNumber);

            // Allocate the outgoing frame
            rtnlos::FrameHistogramDataPtr fhd(new rtnlos::FrameHistogramDataType(parsedData->_frameNumber));

            // Binning
            BuildHistogram_OMP(
                parsedData->_indexData.data(), parsedData->_timeData.data(), _frequencies, 
                fhd->_histogram, 
                static_cast<uint32_t>(parsedData->_indexData.size()));

            spdlog::debug("{},{:8.2f} ms to build histogram of {} photons for frame", parsedData->_frameNumber, timer.elapsed().count(), parsedData->_indexData.size());

            // Push histogram data into the outgoing queue
            spdlog::trace("Pushing frame {} to the outgoing queue (queue.size == {})", parsedData->_frameNumber, _outgoingHistograms.Size());
            _outgoingHistograms.Push(fhd);
        }

		spdlog::warn("FrameHistogramBuilder worker thread stopped");
    }

    template<int NINDICES, int NFREQ>
    void FrameHistogramBuilder<NINDICES, NFREQ>::Stop() const
    {
        // There may be a push blocking if the queue is full, so tell the queue to abort it's operation
        _incomingParsed.Abort();
        _outgoingHistograms.Abort();
    }

    template<int NINDICES, int NFREQ>
    void FrameHistogramBuilder<NINDICES, NFREQ>::BuildHistogram_OMP(
        const uint32_t* indexData,
        const float* timeData,
        const float* frequencyData,
        cufftComplex* histData,
        const uint32_t nSamples) const
    {
        // Optimal ordering for cuda rsd input
		cufftComplex* histogram = histData;

        // Note, the -2PI/MaxT coefficient is already in the frequencyData
#pragma omp parallel for
        for (int f = 0; f < NFREQ; f++) 
        {
            for (uint32_t s = 0; s < nSamples; s++) 
            {
#if USE_FDH_LOOKUP_TABLE
                int idx = static_cast<int>(timeData[s] + 0.5f - _photonMinTime) * NFREQ + f;
                histogram[f * NINDICES + indexData[s]].x += _cosLookup[idx];    // std::cos(frequencyData[f] * timeData[s]);
                histogram[f * NINDICES + indexData[s]].y += _sinLookup[idx];    // std::sin(frequencyData[f] * timeData[s]);
#else
                histogram[f][indexData[s]][0] += std::cos(frequencyData[f] * timeData[s]);
                histogram[f][indexData[s]][1] += std::sin(frequencyData[f] * timeData[s]);
#endif
            }
        }
    }

    // Explicit instantiation
    template class FrameHistogramBuilder<NUMBER_OF_ROWS * NUMBER_OF_COLS, NUMBER_OF_FREQUENCIES>;
}
