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

#pragma once

#include "neural/backend.h"
#include "search/classic/node.h"
// Do not include "trainingdata/writer.h" here to avoid circular dependency.
// Forward declare TrainingDataWriter instead.

namespace lczero {

class TrainingDataWriter; // Forward declaration

#pragma pack(push, 1)

// V7 Training Data Structure
// Based on V6, with additional fields as per chunkparser.py analysis.
// Total size: 8396 bytes.
struct V7TrainingData {
  uint32_t version;                 // Bytes 0-3
  uint32_t input_format;            // Bytes 4-7
  float probabilities[1858];        // Bytes 8-7439 (7432 bytes)
  uint64_t planes[104];             // Bytes 7440-8271 (832 bytes)
  uint8_t castling_us_ooo;          // Byte 8272
  uint8_t castling_us_oo;           // Byte 8273
  uint8_t castling_them_ooo;        // Byte 8274
  uint8_t castling_them_oo;         // Byte 8275
  uint8_t side_to_move_or_enpassant;// Byte 8276
  uint8_t rule50_count;             // Byte 8277
  // Bitfield (see V6 comments for details)
  uint8_t invariance_info;          // Byte 8278
  // Was result in V3/V4, dummy in V5/V6. Retained.
  uint8_t dummy;                    // Byte 8279
  float root_q;                     // Bytes 8280-8283
  float best_q;                     // Bytes 8284-8287
  float root_d;                     // Bytes 8288-8291
  float best_d;                     // Bytes 8292-8295
  float root_m;                     // Bytes 8296-8299 (In plies)
  float best_m;                     // Bytes 8300-8303 (In plies)
  float plies_left;                 // Bytes 8304-8307 (Training target for MLH)
  float result_q;                   // Bytes 8308-8311
  float result_d;                   // Bytes 8312-8315
  float played_q;                   // Bytes 8316-8319
  float played_d;                   // Bytes 8320-8323
  float played_m;                   // Bytes 8324-8327
  // The following may be NaN if not found in cache.
  float orig_q;                     // Bytes 8328-8331 (For value repair)
  float orig_d;                     // Bytes 8332-8335
  float orig_m;                     // Bytes 8336-8339
  uint32_t visits;                  // Bytes 8340-8343
  // Indices in the probabilities array.
  uint16_t played_idx;              // Bytes 8344-8345
  uint16_t best_idx;                // Bytes 8346-8347
  // Kullback-Leibler divergence
  float policy_kld;                 // Bytes 8348-8351
  // V7 specific fields start here.
  // `st_q` effectively replaces V6's `reserved` (uint32_t) field, now a float.
  float st_q;                       // Bytes 8352-8355 (Short-term Q EMA)
  float st_d;                       // Bytes 8356-8359 (Short-term D EMA)
  uint16_t opp_played_idx;          // Bytes 8360-8361 (Opponent's played move index in next state)
  uint16_t next_played_idx;         // Bytes 8362-8363 (Our played move index in state after opponent's move)
  float extra[8];                   // Bytes 8364-8395 (Reserved for future use, 32 bytes)
} PACKED_STRUCT;
static_assert(sizeof(V7TrainingData) == 8396, "Wrong struct size for V7TrainingData");

#pragma pack(pop)

class V7TrainingDataArray {
 public:
  V7TrainingDataArray(FillEmptyHistory white_fill_empty_history,
                      FillEmptyHistory black_fill_empty_history,
                      pblczero::NetworkFormat::InputFormat input_format);

  // Add a chunk.
  void Add(const classic::Node* node, const PositionHistory& history,
           classic::Eval best_eval, classic::Eval played_eval,
           bool best_is_proven, Move best_move, Move played_move,
           std::span<Move> legal_moves,
           const std::optional<EvalResult>& nneval, float policy_softmax_temp);

  // Writes training data to a file.
  void Write(TrainingDataWriter* writer, GameResult result,
             bool adjudicated) const;

 private:
  std::vector<V7TrainingData> training_data_;
  FillEmptyHistory fill_empty_history_[2];
  pblczero::NetworkFormat::InputFormat input_format_;
};

}  // namespace lczero
