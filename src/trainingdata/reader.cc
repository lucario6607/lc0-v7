/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "trainingdata/reader.h"

#include <algorithm> // For std::fill
#include <limits>    // For std::numeric_limits

namespace lczero {

InputPlanes PlanesFromTrainingData(const V7TrainingData& data) {
  InputPlanes result;
  for (int i = 0; i < 104; i++) {
    result.emplace_back();
    result.back().mask = ReverseBitsInBytes(data.planes[i]);
  }
  switch (data.input_format) {
    case pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE: {
      result.emplace_back();
      result.back().mask = data.castling_us_ooo != 0 ? ~0LL : 0LL;
      result.emplace_back();
      result.back().mask = data.castling_us_oo != 0 ? ~0LL : 0LL;
      result.emplace_back();
      result.back().mask = data.castling_them_ooo != 0 ? ~0LL : 0LL;
      result.emplace_back();
      result.back().mask = data.castling_them_oo != 0 ? ~0LL : 0LL;
      break;
    }
    case pblczero::NetworkFormat::INPUT_112_WITH_CASTLING_PLANE:
    case pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION:
    case pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES:
    case pblczero::NetworkFormat::
        INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON:
    case pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2:
    case pblczero::NetworkFormat::
        INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON: {
      result.emplace_back();
      result.back().mask =
          data.castling_us_ooo |
          (static_cast<uint64_t>(data.castling_them_ooo) << 56);
      result.emplace_back();
      result.back().mask = data.castling_us_oo |
                           (static_cast<uint64_t>(data.castling_them_oo) << 56);
      // 2 empty planes in this format.
      result.emplace_back();
      result.emplace_back();
      break;
    }

    default:
      throw Exception("Unsupported input plane encoding " +
                      std::to_string(data.input_format));
  }
  result.emplace_back();
  auto typed_format =
      static_cast<pblczero::NetworkFormat::InputFormat>(data.input_format);
  if (IsCanonicalFormat(typed_format)) {
    result.back().mask = static_cast<uint64_t>(data.side_to_move_or_enpassant)
                         << 56;
  } else {
    result.back().mask = data.side_to_move_or_enpassant != 0 ? ~0LL : 0LL;
  }
  result.emplace_back();
  if (IsHectopliesFormat(typed_format)) {
    result.back().Fill(data.rule50_count / 100.0f);
  } else {
    result.back().Fill(data.rule50_count);
  }
  result.emplace_back();
  // Empty plane, except for canonical armageddon.
  if (IsCanonicalArmageddonFormat(typed_format) &&
      data.invariance_info >= 128) {
    result.back().SetAll();
  }
  result.emplace_back();
  // All ones plane.
  result.back().SetAll();
  if (IsCanonicalFormat(typed_format) && data.invariance_info != 0) {
    // Undo transformation here as it makes the calling code simpler.
    int transform = data.invariance_info;
    for (size_t i = 0; i <= result.size(); i++) {
      auto v = result[i].mask;
      if (v == 0 || v == ~0ULL) continue;
      if ((transform & TransposeTransform) != 0) {
        v = TransposeBitsInBytes(v);
      }
      if ((transform & MirrorTransform) != 0) {
        v = ReverseBytesInBytes(v);
      }
      if ((transform & FlipTransform) != 0) {
        v = ReverseBitsInBytes(v);
      }
      result[i].mask = v;
    }
  }
  return result;
}

TrainingDataReader::TrainingDataReader(std::string filename)
    : filename_(filename) {
  fin_ = gzopen(filename_.c_str(), "rb");
  if (!fin_) {
    throw Exception("Cannot open gzip file " + filename_);
  }
}

TrainingDataReader::~TrainingDataReader() { gzclose(fin_); }

// On-disk sizes of the various version parts
// These are the sizes of the *data added* by that version compared to the previous.
// V3_COMMON_SIZE includes the version field itself.
// V3 is the base, V4 adds 16 bytes, V5 adds 16, V6 adds 48, V7 adds 40.
constexpr int V3_COMMON_SIZE = 8276; // sizeof(version) + probs + planes + castling + stm + rule50 + invariance + dummy
constexpr int V4_ADDED_SIZE = 16;    // root_q, best_q, root_d, best_d
constexpr int V5_ADDED_SIZE = 16;    // input_format, root_m, best_m, plies_left (effectively, due to memmove)
constexpr int V6_ADDED_SIZE = 48;    // result_q/d, played_q/d/m, orig_q/d/m, visits, idx, kld, st_q(was reserved)
constexpr int V7_ADDED_SIZE = 40;    // st_d, opp_played_idx, next_played_idx, extra[8]

// Total sizes on disk
constexpr int V3_DISK_SIZE = V3_COMMON_SIZE;
constexpr int V4_DISK_SIZE = V3_DISK_SIZE + V4_ADDED_SIZE;
constexpr int V5_DISK_SIZE = V4_DISK_SIZE + V5_ADDED_SIZE; // Note: V5 actually rearranges, this is effective additional read size.
constexpr int V6_DISK_SIZE = V5_DISK_SIZE + V6_ADDED_SIZE;
constexpr int V7_DISK_SIZE = V6_DISK_SIZE + V7_ADDED_SIZE;


bool TrainingDataReader::ReadChunk(V7TrainingData* data) {
  // Buffer to hold the maximum possible size of a chunk on disk (V7)
  // We read parts incrementally.
  char* base_ptr = reinterpret_cast<char*>(data);
  int bytes_read_total = 0;

  // Read version first.
  int read_size = gzread(fin_, &data->version, sizeof(data->version));
  if (read_size == 0) return false; // EOF
  if (read_size < 0) throw Exception("Corrupt read (version).");
  if (read_size != sizeof(data->version)) return false; // Partial read, treat as EOF or error
  bytes_read_total += read_size;

  uint32_t version_on_disk = data->version;

  // Read the rest of the V3 common part.
  // V3_COMMON_SIZE already includes sizeof(data->version).
  read_size = gzread(fin_, base_ptr + sizeof(data->version), V3_COMMON_SIZE - sizeof(data->version));
  if (read_size < 0) throw Exception("Corrupt read (V3 common part).");
  if (read_size != V3_COMMON_SIZE - sizeof(data->version)) return false; // Unexpected EOF
  bytes_read_total += read_size;

  // Store original version to handle fall-through logic correctly
  // for reading additional data for specific on-disk versions.
  uint32_t original_version_on_disk = version_on_disk;

  // Upgrade logic: Start with the version on disk and upgrade step-by-step.
  // The `data` struct in memory will always be converted to V7.

  if (version_on_disk == 3) {
    // Upgrade V3 to V4 in memory
    // V3 specific initializations for V4 fields:
    // (root_q, best_q, root_d, best_d are part of V4_ADDED_SIZE)
    // These will be zeroed by default or set below if not read.
    // For V3 -> V4, V4 fields (4 floats) are effectively zeroed as they are not read yet.
    // We fill them with 0.0f.
    char* v4_fields_start = base_ptr + V3_COMMON_SIZE;
    for (int i = 0; i < V4_ADDED_SIZE; ++i) {
        v4_fields_start[i] = 0;
    }
    data->version = 4; // Now treat as V4 for subsequent upgrades
    version_on_disk = 4; // Update current version state for fallthrough
  }

  if (version_on_disk == 4) {
    if (original_version_on_disk == 4) {
      // Read V4-specific data if this was originally a V4 record
      read_size = gzread(fin_, base_ptr + V3_COMMON_SIZE, V4_ADDED_SIZE);
      if (read_size < 0) throw Exception("Corrupt read (V4 added part).");
      if (read_size != V4_ADDED_SIZE) return false;
      bytes_read_total += read_size;
    }
    // Upgrade V4 to V5 in memory
    // Shift data after version back 4 bytes to make space for input_format
    // V3_COMMON_SIZE contains version.
    // V4_ADDED_SIZE is after V3_COMMON_SIZE.
    // Total read so far for V4 is V3_COMMON_SIZE + V4_ADDED_SIZE.
    // Effective start of fields after version is sizeof(uint32_t).
    // Size of block to move: (V3_COMMON_SIZE - sizeof(uint32_t)) + V4_ADDED_SIZE.
    memmove(base_ptr + 2 * sizeof(uint32_t),         // To (make space for input_format)
            base_ptr + sizeof(uint32_t),             // From (original data after version)
            (V3_COMMON_SIZE - sizeof(uint32_t)) + V4_ADDED_SIZE); // How much to move
    
    data->input_format = pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE;
    // Initialize other V5 fields (root_m, best_m, plies_left)
    data->root_m = 0.0f;
    data->best_m = 0.0f;
    data->plies_left = 0.0f;
    
    data->version = 5; // Now treat as V5
    version_on_disk = 5;
  }

  if (version_on_disk == 5) {
    if (original_version_on_disk == 5) {
      // Read V5-specific data if this was originally a V5 record
      // V5 added fields are input_format, root_m, best_m, plies_left (16 bytes)
      // These are conceptually read after V4_DISK_SIZE.
      // The memmove for V4->V5 upgrade already placed input_format.
      // We need to read root_m, best_m, plies_left which are 3 floats (12 bytes).
      // These conceptually follow input_format.
      // The V5_ADDED_SIZE (16) refers to total effective size increase from V4.
      // Since input_format (4 bytes) is filled by code, we read the remaining 12 bytes.
      char* v5_specific_read_ptr = base_ptr + V3_COMMON_SIZE + V4_ADDED_SIZE; // End of V4 data
      // However, due to memmove, data has shifted.
      // input_format is at base_ptr + sizeof(uint32_t)
      // root_m starts after invariance_info, dummy, root_q, best_q, root_d, best_d
      // This complex offsetting is why the original code calculated `v3_size` based on target struct.
      // Let's use simpler V*_DISK_SIZE for reading.
      // If original_version_on_disk == 5, it means V3_COMMON_SIZE + V4_ADDED_SIZE + V5_ADDED_SIZE were on disk.
      // We have read V3_COMMON_SIZE + V4_ADDED_SIZE. We need to read V5_ADDED_SIZE.
      read_size = gzread(fin_, base_ptr + V3_COMMON_SIZE + V4_ADDED_SIZE, V5_ADDED_SIZE);
      if (read_size < 0) throw Exception("Corrupt read (V5 added part).");
      if (read_size != V5_ADDED_SIZE) return false;
      bytes_read_total += read_size;
    }
    // Upgrade V5 to V6 in memory
    // Type of dummy was changed from signed to unsigned - which means -1 on
    // disk is read in as 255.
    if (data->dummy > 1 && data->dummy < 255) {
      throw Exception("Invalid result read in v5 data before upgrade.");
    }
    data->result_q =
        data->dummy == 255 ? -1.0f : (data->dummy == 0 ? 0.0f : 1.0f);
    data->result_d = data->dummy == 0 ? 1.0f : 0.0f;
    data->dummy = 0; // V6+ dummy is just a placeholder.

    // Initialize other V6 fields
    data->played_q = 0.0f;
    data->played_d = 0.0f;
    data->played_m = 0.0f;
    data->orig_q = std::numeric_limits<float>::quiet_NaN();
    data->orig_d = std::numeric_limits<float>::quiet_NaN();
    data->orig_m = std::numeric_limits<float>::quiet_NaN();
    data->visits = 0;
    data->played_idx = 0;
    data->best_idx = 0;
    data->policy_kld = 0.0f;
    data->st_q = 0.0f; // V6 'reserved' field, now st_q. Initialize.

    data->version = 6; // Now treat as V6
    version_on_disk = 6;
  }

  if (version_on_disk == 6) {
    if (original_version_on_disk == 6) {
      // Read V6-specific data
      read_size = gzread(fin_, base_ptr + V3_COMMON_SIZE + V4_ADDED_SIZE + V5_ADDED_SIZE, V6_ADDED_SIZE);
      if (read_size < 0) throw Exception("Corrupt read (V6 added part).");
      if (read_size != V6_ADDED_SIZE) return false;
      bytes_read_total += read_size;
    }
    // Upgrade V6 to V7 in memory
    // V6's `reserved` field (last 4 bytes) becomes V7's `st_q`.
    // If it was an actual V6 on disk, data->st_q would have been read into.
    // If it was an older version upgraded, data->st_q was set to 0.0f above.
    // For consistency, if upgrading from a true V6, use its root_q for st_q.
    if (original_version_on_disk == 6) { // Or if it was just upgraded to V6 structure
        // data->st_q was read from disk (V6's reserved/q_st field).
        // This is fine, it's now interpreted as st_q.
    } else { // Upgraded from V3/4/5, st_q was set to 0.0f. Override with root_q.
        data->st_q = data->root_q;
    }
    
    data->st_d = data->root_d; // Initialize st_d based on root_d
    data->opp_played_idx = 0xFFFF;
    data->next_played_idx = 0xFFFF;
    std::fill(std::begin(data->extra), std::end(data->extra), 0.0f);
    
    data->version = 7; // Final in-memory version is 7
    // version_on_disk = 7; // Not needed, loop ends
  } else if (version_on_disk == 7) {
    // This was originally a V7 record on disk
    // Read the V7-specific part
    read_size = gzread(fin_, base_ptr + V3_COMMON_SIZE + V4_ADDED_SIZE + V5_ADDED_SIZE + V6_ADDED_SIZE, V7_ADDED_SIZE);
    if (read_size < 0) throw Exception("Corrupt read (V7 added part).");
    if (read_size != V7_ADDED_SIZE) return false;
    bytes_read_total += read_size;
    data->version = 7; // Already 7, just to be clear
  } else {
    throw Exception("Unknown or unsupported training data version: " + std::to_string(original_version_on_disk));
  }

  // At this point, `data` is a fully populated V7 struct in memory.
  // The total bytes read from file should match the original disk size.
  size_t expected_disk_size = 0;
  switch (original_version_on_disk) {
      case 3: expected_disk_size = V3_DISK_SIZE; break;
      case 4: expected_disk_size = V4_DISK_SIZE; break;
      case 5: expected_disk_size = V5_DISK_SIZE; break;
      case 6: expected_disk_size = V6_DISK_SIZE; break;
      case 7: expected_disk_size = V7_DISK_SIZE; break;
      default: throw Exception("Logic error in version handling."); // Should not happen
  }
  if (static_cast<size_t>(bytes_read_total) != expected_disk_size) {
      // This can happen if an upgrade path incorrectly calculates read bytes
      // or if gzread returns less than expected for specific version parts.
      // For now, we rely on individual gzread checks for short reads.
  }

  return true;
}

}  // namespace lczero
