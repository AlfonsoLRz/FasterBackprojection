#include "stdafx.h"

#include "opencv4/opencv2/core/persistence.hpp"
#include "SceneParameters.h"

namespace rtnlos
{
	bool FileExists(const std::string& name)
	{
		std::ifstream f(name.c_str());
		return f.is_open();
	}

	// helper to read scalars from cv::FileStorage
	template<class T>
	T ReadFromFS(const cv::FileStorage& fs, const std::string& name)
	{
		cv::Mat val;
		fs[name] >> val;
		if (val.rows != 1 || val.cols != 1) {
			std::stringstream ss;
			ss << "Value '" << name << "' cannot be read as a scalar from the file";
			throw std::logic_error(ss.str());
		}
		// in the file stores that we use, all values are written as floats
		return (T)val.at<float>(0, 0);
	}

	// specialization of helper to read 1D-arrays from cv::FileStorage
	template<class D>
	std::vector<D> ReadVecFromFS(const cv::FileStorage& fs, const std::string& name)
	{
		cv::Mat val;
		fs[name] >> val;
		// either rows or columns must be 1 (this function only reads vectors)
		if (val.rows != 1 && val.cols != 1) {
			std::stringstream ss;
			ss << "Value '" << name << "' cannot be read as a vector from the file";
			throw std::logic_error(ss.str());
		}
		float* ptr = reinterpret_cast<float*>(val.data);
		std::vector<D> ret(std::max(val.cols, val.rows));
		for (auto i = 0; i < ret.size(); i++)
			ret[i] = static_cast<D>(ptr[i]);
		return ret;
	}

	void SceneParameters::Initialize(const std::string& filename)
	{
		if (!FileExists(filename)) {
			std::stringstream ss;
			ss << "Invalid scene parameters file ('" << filename << "')" << std::endl;
			throw std::logic_error(ss.str());
		}

		// Transfer data from xml to cvMat
		cv::FileStorage fs(filename, cv::FileStorage::READ);

		_numComponents			= ReadFromFS<int>(fs, "num_component");
		_apertureWidth			= ReadFromFS<int>(fs, "apt_width");
		_apertureHeight			= ReadFromFS<int>(fs, "apt_height");
		_skipRate				= ReadFromFS<int>(fs, "param_skipRate");
		_spadIDs				= ReadFromFS<int>(fs, "SpadIDs");
		_syncRate				= ReadFromFS<float>(fs, "param_syncrate");
		_galvoRate				= ReadFromFS<float>(fs, "param_galvoSamplingRate");
		_centeringCoeff			= ReadFromFS<int>(fs, "centering_coef");
		_binZeros				= ReadVecFromFS<uint32_t>(fs, "param_bin_zero");
		_binLength				= ReadFromFS<uint32_t>(fs, "param_single_histoBinLength");
		_deltaT					= ReadFromFS<float>(fs, "param_deltat");
		_downsamplingRate		= ReadFromFS<int>(fs, "ts_ds_rate");
		_zGate					= ReadFromFS<float>(fs, "z_gate_base");
		_resolution				= ReadFromFS<float>(fs, "param_resolution");
		_depthMin				= ReadFromFS<float>(fs, "tag_mindepth");
		_depthMax				= ReadFromFS<float>(fs, "tag_maxdepth");
		_depthDelta				= ReadFromFS<float>(fs, "tag_deltadepth");
		_depthOffset			= ReadFromFS<float>(fs, "tag_depthoffset");
		_samplingSpacing		= ReadFromFS<float>(fs, "sampling_spacing");
		_weights				= ReadVecFromFS<float>(fs, "weight");
		_lambdas				= ReadVecFromFS<float>(fs, "lambda_loop");
		_omegas					= ReadVecFromFS<float>(fs, "omega_space");
		_freMask				= ReadVecFromFS<float>(fs, "fre_mask");
		_apertureFullSize		= ReadVecFromFS<float>(fs, "aperturefullsize");
		_d1						= ReadVecFromFS<float>(fs, "param_D1");

		for (int i = 0; i < _spadIDs; i++) {
			std::string spad_id = std::to_string(i + 1);
			_spadIndex.emplace_back(ReadVecFromFS<float>(fs, "param_spad_ind_" + spad_id));
			_offset.emplace_back(ReadVecFromFS<float>(fs, "offset_" + spad_id));
			_t0Gated.emplace_back(ReadVecFromFS<float>(fs, "t0_" + spad_id));
			_d4.emplace_back(ReadVecFromFS<float>(fs, "param_D4_" + spad_id));
		}
	};
}
