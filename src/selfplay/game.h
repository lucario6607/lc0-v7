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

#include "chess/pgn.h"
#include "chess/position.h"
#include "chess/uciloop.h"
#include "neural/backend.h"
#include "search/classic/search.h"
#include "search/classic/stoppers/stoppers.h"
#include "trainingdata/trainingdata.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace classic {
class Node;
}  // namespace classic

// Contains the state of a game (both training and match game).
class Game {
 public:
  // Game constructor takes an PGNBuilder object that is used to store a PGN of
  // the game as it's being played. It is expected that the `pgn_builder` object
  // outlives the game object.
  //
  // It also takes an optional `backend`, which, if supplied, will be used to
  // get NNCache data, which will be recorded in the training data.
  //
  // Also takes `training_data_options`, which are used to customize behavior
  // for writing training data.
  Game(int id, PGNBuilder* pgn_builder, neural::Network* backend,
       const TrainingDataOptions& training_data_options,
       const FillEmptyHistory& white_fill_empty_history,
       const FillEmptyHistory& black_fill_empty_history);
  ~Game();

  // Gets the underlying position history.
  const PositionHistory& GetHistory() const { return history_; }
  PositionHistory& GetHistory() { return history_; }

  // Resets the game to the given FEN string.
  void Reset(const std::string& fen);

  // Adds the best move to the game history.
  bool AddMove(Move move, classic::Eval best_eval, classic::Eval played_eval,
               bool best_is_proven, const classic::Node* node,
               float policy_softmax_temp, std::span<Move> legal_moves,
               const std::optional<EvalResult>& nneval, std::string comment);

  // Gets the game result, considering the last position.
  GameResult GetResult(bool adjudicated) const;

  // Writes training data. Returns path to training file.
  std::string WriteTrainingData(GameResult result, bool adjudicated);

  // Gets PGN for the game.
  std::string GetPGN() const;

  // Checks whether the game should be stopped. Returns the reason.
  GameResultReason CheckStop(const SelfplayOptions& options, int max_plies);

  // Checks whether the game should be drawn.
  DrawReason CheckDraw(const SelfplayOptions& options);

  // Checks whether the game should be resigned. Returns true if it should.
  bool CheckResign(const SelfplayOptions& options,
                   const classic::Node* root_node, bool is_our_turn);

  // Checks whether the game is over.
  bool IsGameOver() const { return is_game_over_; }
  void SetGameOver() { is_game_over_ = true; }

  // Gets the ID of this game.
  int GetId() const { return id_; }

  // Get number of plies played in this game.
  size_t GetPgnPly() const { return history_.GetPgnPly(); }

  // Total moves in game including variations.
  size_t GetTotalPly() const { return history_.GetLength(); }

  // Was this game an opening book game.
  bool FromBook() const { return from_book_; }
  void SetFromBook() { from_book_ = true; }

 private:
  // Resets game to the given position history.
  void Reset(const PositionHistory& history);

  // Common reset logic between Reset(fen) and Reset(history).
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
