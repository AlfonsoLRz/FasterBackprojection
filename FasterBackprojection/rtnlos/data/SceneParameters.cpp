#include "stdafx.h"

#include "opencv4/opencv2/core/persistence.hpp"
#include "SceneParameters.h"

namespace NLOS {

	bool FileExists(const std::string& name) {
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
		float* ptr = (float*)val.data;
		std::vector<D> ret(std::max(val.cols, val.rows));
		for (auto i = 0; i < ret.size(); i++)
			ret[i] = (D)(ptr[i]);
		return ret;
	}

	void SceneParameters::Initialize(const std::string& filename) {
		if (!FileExists(filename)) {
			std::stringstream ss;
			ss << "Invalid scene parameters file ('" << filename << "')" << std::endl;
			throw std::logic_error(ss.str());
		}

		// Transfer data from xml to cvMat
		cv::FileStorage fs(filename, cv::FileStorage::READ);

		NumComponents	 = ReadFromFS<int>(fs, "num_component");
		AptWidth		 = ReadFromFS<int>(fs, "apt_width");
		AptHeight		 = ReadFromFS<int>(fs, "apt_height");
		SkipRate		 = ReadFromFS<int>(fs, "param_skipRate");
		SpadIDs			 = ReadFromFS<int>(fs, "SpadIDs");
		SyncRate		 = ReadFromFS<float>(fs, "param_syncrate");
		GalvoRate		 = ReadFromFS<float>(fs, "param_galvoSamplingRate");
		CenteringCoeff   = ReadFromFS<int>(fs, "centering_coef");
		BinZeros		 = ReadVecFromFS<uint32_t>(fs, "param_bin_zero");
		BinLength		 = ReadFromFS<uint32_t>(fs, "param_single_histoBinLength");
		DeltaT			 = ReadFromFS<float>(fs, "param_deltat");
		DownsamplingRate = ReadFromFS<int>(fs, "ts_ds_rate");
		Z_Gate			 = ReadFromFS<float>(fs, "z_gate_base");
		Resolution		 = ReadFromFS<float>(fs, "param_resolution");
		DepthMin		 = ReadFromFS<float>(fs, "tag_mindepth");
		DepthMax		 = ReadFromFS<float>(fs, "tag_maxdepth");
		DepthDelta		 = ReadFromFS<float>(fs, "tag_deltadepth");
		DepthOffset		 = ReadFromFS<float>(fs, "tag_depthoffset");
		SamplingSpacing  = ReadFromFS<float>(fs, "sampling_spacing");
		Weights			 = ReadVecFromFS<float>(fs, "weight");
		Lambdas			 = ReadVecFromFS<float>(fs, "lambda_loop");
		Omegas			 = ReadVecFromFS<float>(fs, "omega_space");
		FreMask			 = ReadVecFromFS<float>(fs, "fre_mask");
		ApertureFullSize = ReadVecFromFS<float>(fs, "aperturefullsize");
		D1				 = ReadVecFromFS<float>(fs, "param_D1");
		for (int i = 0; i < SpadIDs; i++) {
			std::string spad_id = std::to_string(i + 1);
			SpadIndex.emplace_back(ReadVecFromFS<float>(fs, "param_spad_ind_" + spad_id));
			Offset.emplace_back(ReadVecFromFS<float>(fs, "offset_" + spad_id));
			T0_Gated.emplace_back(ReadVecFromFS<float>(fs, "t0_" + spad_id));
			D4.emplace_back(ReadVecFromFS<float>(fs, "param_D4_" + spad_id));
		}
	};
}
