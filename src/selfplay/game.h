/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2021 The LCZero Authors

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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "search/classic/search.h"      // Provides Node, NodeTree, Search definitions
#include "trainingdata/trainingdata.h"   // For V7TrainingDataArray
#include "utils/history.h"               // For PositionHistory (ensure this is src/utils/history.h)
#include "chess/pgnbuilder.h"            // For PGNBuilder
#include "utils/selfplayoptions.h"       // For SelfplayOptions
#include "utils/trainingdataoptions.h"   // For TrainingDataOptions
#include "neural/encoder.h"              // For FillEmptyHistory
#include "neural/backend.h"              // For neural::Network, neural::EvalResult (used in Game methods)
#include "chess/position.h"              // For Move, GameResult, DrawReason, GameResultReason (used in Game methods)


namespace lczero {
namespace classic {
class Node; // Forward declaration
struct Eval;  // Forward declaration
}  // namespace classic


// Contains the state of a game (both training and match game).
class Game {
 public:
  Game(int id, PGNBuilder* pgn_builder, neural::Network* backend,
       const TrainingDataOptions& training_data_options,
       const FillEmptyHistory& white_fill_empty_history,
       const FillEmptyHistory& black_fill_empty_history);
  ~Game();

  const PositionHistory& GetHistory() const { return history_; }
  PositionHistory& GetHistory() { return history_; }

  void Reset(const std::string& fen);

  bool AddMove(Move move, classic::Eval best_eval, classic::Eval played_eval,
               bool best_is_proven, const classic::Node* node,
               float policy_softmax_temp, std::span<Move> legal_moves,
               const std::optional<neural::EvalResult>& nneval, std::string comment);

  GameResult GetResult(bool adjudicated) const;
  std::string WriteTrainingData(GameResult result, bool adjudicated);
  std::string GetPGN() const;

  GameResultReason CheckStop(const SelfplayOptions& options, int max_plies);
  DrawReason CheckDraw(const SelfplayOptions& options);
  bool CheckResign(const SelfplayOptions& options,
                   const classic::Node* root_node, bool is_our_turn);

  bool IsGameOver() const { return is_game_over_; }
  void SetGameOver() { is_game_over_ = true; }
  int GetId() const { return id_; }
  size_t GetPgnPly() const { return history_.GetPgnPly(); }
  size_t GetTotalPly() const { return history_.GetLength(); }
  bool FromBook() const { return from_book_; }
  void SetFromBook() { from_book_ = true; }

 private:
  void Reset(const PositionHistory& history);
  void DoReset();

  const int id_;
  bool is_game_over_ = false;
  bool from_book_ = false;
  PositionHistory history_;
  PGNBuilder* const pgn_builder_;
  neural::Network* const backend_ = nullptr;
  TrainingDataOptions training_data_options_;
  V7TrainingDataArray training_data_;
};

}  // namespace lczero
