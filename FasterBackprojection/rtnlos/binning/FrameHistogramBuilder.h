#pragma once

#include "data/SensorData.h"
#include "data/SceneParameters.h"
#include "NlosDataProcessor.h"

#define USE_FDH_LOOKUP_TABLE 1

namespace rtnlos
{
    // Stage 2: parse the raw data and bin it into a Fourier domain histogram.
    template<int NINDICES, int NFREQ>
    class FrameHistogramBuilder : public rtnlos::NlosDataProcessor
	{
    private:
        ParsedSensorDataQueue&      _incomingParsed;
        FrameHistogramDataQueue&    _outgoingHistograms;
        bool                        _frequencyMajorOrder = true;

        float                       _frequencies[NFREQ];

    public:
        FrameHistogramBuilder(rtnlos::ParsedSensorDataQueue& incoming, rtnlos::FrameHistogramDataQueue& outgoing)
	        :   _incomingParsed(incoming), _outgoingHistograms(outgoing), _frequencies{}, _photonMinTime(0),
    			_photonMaxTime(0),
				_numTimes(0)
        {
        }

        void Initialize(const rtnlos::SceneParameters& sceneParameters);
		void DoWork();
        void Stop() const;
		void setFrequencyMajorOrder(bool order) { _frequencyMajorOrder = order; }

    private:
        void Work() const;
        void BuildHistogram(
            const uint32_t* indexData,
            const float* timeData,
            const float* frequencyData,
            cufftComplex* histData,
            const uint32_t nSamples) const;

#if USE_FDH_LOOKUP_TABLE
        int _photonMinTime, _photonMaxTime, _numTimes;
        std::vector<float>  _sinLookup, _cosLookup;

        void BuildLookupTables();
#endif
    };
}
