#pragma once

namespace NLOS {
	union T3Rec {
		uint32_t allbits;
		struct {
			unsigned nsync : 10;

			// last 10 bits, number of sync period
			// this is 2 things:
			// a : for an overflow it's how many overflows since last
			// photon / marker
			// b : for photon / marker it's how many syncs since last overflow
			// | | | | | | | | | | | | | | | | | | | | | | |x|x|x|x|x|x|x|x|x|x|

			unsigned dtime : 15;

			// next 15 bits, delay from last sync in units of chosen resolution
			// the dtime unit depends on "Resolution" that can be obtained from header
			// DTime: Arrival time(units) of Photon after last Sync event
			// DTime* Resolution = Real time arrival of Photon after last Sync event
			// | | | | | | | |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x| | | | | | | | | | |

			unsigned channel : 6;

			// next 6 bits, for photons, channel 0~7, for overflows, channel = 63, for markers, special = 1, channel 1~4
			//| |x|x|x|x|x|x| | | | | | | | | | | | | | | | | | | | | | | | | |

			unsigned special : 1;

			// first bit: special = 1 indicates marker or overflows
		} bits;
	};
}
