/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2024 The LCZero Authors

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

#include "trainingdata/rescorer.h"

#include <optional>
#include <sstream>
#include <algorithm> // For std::find

#include "gtb-probe.h"
#include "neural/decoder.h"
#include "syzygy/syzygy.h"
#include "trainingdata/reader.h"
#include "utils/filesystem.h"
#include "utils/optionsparser.h"

namespace lczero {

namespace {
const OptionId kSyzygyTablebaseId{"syzygy-paths", "",
                                  "List of Syzygy tablebase directories"};
const OptionId kGaviotaTablebaseId{"gaviotatb-paths", "",
                                   "List of Gaviota tablebase directories"};
const OptionId kInputDirId{
    "input", "", "Directory with gzipped files in need of rescoring."};
const OptionId kPolicySubsDirId{"policy-substitutions", "",
                                "Directory with gzipped files are to use to "
                                "replace policy for some of the data."};
const OptionId kOutputDirId{"output", "", "Directory to write rescored files."};
const OptionId kThreadsId{"threads", "",
                          "Number of concurrent threads to rescore with.", 't'};
const OptionId kTempId{"temperature", "",
                       "Additional temperature to apply to policy target."};
const OptionId kDistributionOffsetId{
    "dist_offset", "",
    "Additional offset to apply to policy target before temperature."};
const OptionId kMinDTZBoostId{
    "dtz_policy_boost", "",
    "Additional offset to apply to policy target before temperature for moves "
    "that are best dtz option."};
const OptionId kNewInputFormatId{
    "new-input-format", "",
    "Input format to convert training data to during rescoring."};
const OptionId kDeblunder{
    "deblunder", "",
    "If true, whether to use move Q information to infer a different Z value "
    "if the the selected move appears to be a blunder."};
const OptionId kDeblunderQBlunderThreshold{
    "deblunder-q-blunder-threshold", "",
    "The amount Q of played move needs to be worse than best move in order to "
    "assume the played move is a blunder."};
const OptionId kDeblunderQBlunderWidth{
    "deblunder-q-blunder-width", "",
    "Width of the transition between accepted temp moves and blunders."};
const OptionId kNnuePlainFileId{"nnue-plain-file", "",
                                "Append SF plain format training data to this "
                                "file. Will be generated if not there."};
const OptionId kNnueBestScoreId{"nnue-best-score", "",
                                "For the SF training data use the score of the "
                                "best move instead of the played one."};
const OptionId kNnueBestMoveId{
    "nnue-best-move", "",
    "For the SF training data record the best move instead of the played one. "
    "If set to true the generated files do not compress well."};
const OptionId kDeleteFilesId{"delete-files", "",
                              "Delete the input files after processing."};

class PolicySubNode {
 public:
  PolicySubNode() {
    for (int i = 0; i < 1858; i++) children[i] = nullptr;
  }
  bool active = false;
  float policy[1858];
  PolicySubNode* children[1858];
};

std::atomic<int> games(0);
std::atomic<int> positions(0);
std::atomic<int> rescored(0);
std::atomic<int> delta(0);
std::atomic<int> rescored2(0);
std::atomic<int> rescored3(0);
std::atomic<int> blunders(0);
std::atomic<int> orig_counts[3];
std::atomic<int> fixed_counts[3];
std::atomic<int> policy_bump(0);
std::atomic<int> policy_nobump_total_hist[11];
std::atomic<int> policy_bump_total_hist[11];
std::atomic<int> policy_dtm_bump(0);
std::atomic<int> gaviota_dtm_rescores(0);
std::map<uint64_t, PolicySubNode> policy_subs;
bool gaviotaEnabled = false;
bool deblunderEnabled = false;
float deblunderQBlunderThreshold = 2.0f;
float deblunderQBlunderWidth = 0.0f;

void DataAssert(bool check_result) {
  if (!check_result) throw Exception("Range Violation");
}

void Validate(const std::vector<V7TrainingData>& fileContents) {
  if (fileContents.empty()) throw Exception("Empty File");

  for (size_t i = 0; i < fileContents.size(); i++) {
    auto& data = fileContents[i];
    DataAssert(
        data.input_format ==
            pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE ||
        data.input_format ==
            pblczero::NetworkFormat::INPUT_112_WITH_CASTLING_PLANE ||
        data.input_format ==
            pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION ||
        data.input_format == pblczero::NetworkFormat::
                                 INPUT_112_WITH_CANONICALIZATION_HECTOPLIES ||
        data.input_format ==
            pblczero::NetworkFormat::
                INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON ||
        data.input_format ==
            pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2 ||
        data.input_format == pblczero::NetworkFormat::
                                 INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON);
    DataAssert(data.best_d >= 0.0f && data.best_d <= 1.0f);
    DataAssert(data.root_d >= 0.0f && data.root_d <= 1.0f);
    DataAssert(data.best_q >= -1.0f && data.best_q <= 1.0f);
    DataAssert(data.root_q >= -1.0f && data.root_q <= 1.0f);
    DataAssert(data.root_m >= 0.0f);
    DataAssert(data.best_m >= 0.0f);
    DataAssert(data.plies_left >= 0.0f);
    switch (data.input_format) {
      case pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE:
        DataAssert(data.castling_them_oo <= 1);
        DataAssert(data.castling_them_ooo <= 1);
        DataAssert(data.castling_us_oo <= 1);
        DataAssert(data.castling_us_ooo <= 1);
        break;
      default:
        // Verifiy at most one bit set.
        DataAssert((data.castling_them_oo & (data.castling_them_oo - 1)) == 0);
        DataAssert((data.castling_them_ooo & (data.castling_them_ooo - 1)) ==
                   0);
        DataAssert((data.castling_us_oo & (data.castling_us_oo - 1)) == 0);
        DataAssert((data.castling_us_ooo & (data.castling_us_ooo - 1)) == 0);
    }
    if (IsCanonicalFormat(static_cast<pblczero::NetworkFormat::InputFormat>(
            data.input_format))) {
      // At most one en-passant bit.
      DataAssert((data.side_to_move_or_enpassant &
                  (data.side_to_move_or_enpassant - 1)) == 0);
    } else {
      DataAssert(data.side_to_move_or_enpassant <= 1);
    }
    DataAssert(data.result_q >= -1 && data.result_q <= 1);
    // Note: result_d validation was "data.result_q <= 1", likely a typo for data.result_d
    DataAssert(data.result_d >= 0 && data.result_d <= 1);
    DataAssert(data.rule50_count <= 100);
    float sum = 0.0f;
    for (size_t j = 0; j < sizeof(data.probabilities) / sizeof(float); j++) {
      float prob = data.probabilities[j];
      DataAssert((prob >= 0.0f && prob <= 1.0f) || prob == -1.0f ||
                 std::isnan(prob));
      if (prob >= 0.0f) {
        sum += prob;
      }
      // Only check best_idx/played_idx for real v6/v7 data (visits > 0 implies not upgraded from ancient format).
      if (data.visits > 0) {
        // Best_idx and played_idx must be marked legal in probabilities.
        if (j == data.best_idx || j == data.played_idx) {
          DataAssert(prob >= 0.0f);
        }
      }
    }
    if (sum < 0.99f || sum > 1.01f) {
      throw Exception("Probability sum error is huge!");
    }
    DataAssert(data.best_idx <= 1858); // Max NN index
    DataAssert(data.played_idx <= 1858); // Max NN index
    DataAssert(data.played_q >= -1.0f && data.played_q <= 1.0f);
    DataAssert(data.played_d >= 0.0f && data.played_d <= 1.0f);
    DataAssert(data.played_m >= 0.0f);
    DataAssert(std::isnan(data.orig_q) ||
               (data.orig_q >= -1.0f && data.orig_q <= 1.0f));
    DataAssert(std::isnan(data.orig_d) ||
               (data.orig_d >= 0.0f && data.orig_d <= 1.0f));
    DataAssert(std::isnan(data.orig_m) || data.orig_m >= 0.0f);
    // V7 specific fields
    // data.st_q might be NaN if upgraded from very old format where root_q was NaN.
    DataAssert(std::isnan(data.st_q) || (data.st_q >= -1.0f && data.st_q <= 1.0f));
    DataAssert(std::isnan(data.st_d) || (data.st_d >= 0.0f && data.st_d <= 1.0f));
    DataAssert(data.opp_played_idx <= 0xFFFF); // Can be 0xFFFF sentinel
    DataAssert(data.next_played_idx <= 0xFFFF); // Can be 0xFFFF sentinel
    // No specific validation for data.extra yet.
  }
}

void Validate(const std::vector<V7TrainingData>& fileContents,
              const MoveList& moves) {
  PositionHistory history;
  int rule50ply;
  int gameply;
  ChessBoard board;
  auto input_format = static_cast<pblczero::NetworkFormat::InputFormat>(
      fileContents[0].input_format);
  PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]), &board,
                &rule50ply, &gameply);
  history.Reset(board, rule50ply, gameply);
  for (size_t i = 0; i < moves.size(); i++) {
    int transform = TransformForPosition(input_format, history);
    // If real v6/v7 data, can confirm that played_idx matches the inferred move.
    if (fileContents[i].visits > 0) {
      if (fileContents[i].played_idx != MoveToNNIndex(moves[i], transform)) {
        throw Exception("Move performed is not listed as played.");
      }
    }
    // Move shouldn't be marked illegal unless there is 0 visits, which should
    // only happen if invariance_info is marked with the placeholder bit (bit 6 / value 64).
    if (!(fileContents[i].probabilities[MoveToNNIndex(moves[i], transform)] >=
          0.0f) &&
        (fileContents[i].invariance_info & (1u << 6)) == 0) {
      std::cerr << "Illegal move: " << moves[i].ToString(true) << std::endl;
      throw Exception("Move performed is marked illegal in probabilities.");
    }
    auto legal = history.Last().GetBoard().GenerateLegalMoves();
    if (std::find(legal.begin(), legal.end(), moves[i]) == legal.end()) {
      std::cerr << "Illegal move: " << moves[i].ToString(true) << std::endl;
      throw Exception("Move performed is an illegal move.");
    }
    history.Append(moves[i]);
  }
}

void gaviota_tb_probe_hard(const Position& pos, unsigned int& info,
                           unsigned int& dtm) {
  unsigned int wsq[17];
  unsigned int bsq[17];
  unsigned char wpc[17];
  unsigned char bpc[17];

  auto stm = pos.IsBlackToMove() ? tb_BLACK_TO_MOVE : tb_WHITE_TO_MOVE;
  ChessBoard board = pos.GetBoard();
  if (pos.IsBlackToMove()) board.Mirror();
  auto epsq = tb_NOSQUARE;
  for (auto sq : board.en_passant()) {
    // Our internal representation stores en_passant 2 rows away
    // from the actual sq.
    if (sq.rank().idx == 0) {
      epsq = (TB_squares)(sq.as_idx() + 16);
    } else {
      epsq = (TB_squares)(sq.as_idx() - 16);
    }
  }
  int idx = 0;
  for (auto sq : (board.ours() & board.kings())) {
    wsq[idx] = (TB_squares)sq.as_idx();
    wpc[idx] = tb_KING;
    idx++;
  }
  for (auto sq : (board.ours() & board.knights())) {
    wsq[idx] = (TB_squares)sq.as_idx();
    wpc[idx] = tb_KNIGHT;
    idx++;
  }
  for (auto sq : (board.ours() & board.queens())) {
    wsq[idx] = (TB_squares)sq.as_idx();
    wpc[idx] = tb_QUEEN;
    idx++;
  }
  for (auto sq : (board.ours() & board.rooks())) {
    wsq[idx] = (TB_squares)sq.as_idx();
    wpc[idx] = tb_ROOK;
    idx++;
  }
  for (auto sq : (board.ours() & board.bishops())) {
    wsq[idx] = (TB_squares)sq.as_idx();
    wpc[idx] = tb_BISHOP;
    idx++;
  }
  for (auto sq : (board.ours() & board.pawns())) {
    wsq[idx] = (TB_squares)sq.as_idx();
    wpc[idx] = tb_PAWN;
    idx++;
  }
  wsq[idx] = tb_NOSQUARE;
  wpc[idx] = tb_NOPIECE;

  idx = 0;
  for (auto sq : (board.theirs() & board.kings())) {
    bsq[idx] = (TB_squares)sq.as_idx();
    bpc[idx] = tb_KING;
    idx++;
  }
  for (auto sq : (board.theirs() & board.knights())) {
    bsq[idx] = (TB_squares)sq.as_idx();
    bpc[idx] = tb_KNIGHT;
    idx++;
  }
  for (auto sq : (board.theirs() & board.queens())) {
    bsq[idx] = (TB_squares)sq.as_idx();
    bpc[idx] = tb_QUEEN;
    idx++;
  }
  for (auto sq : (board.theirs() & board.rooks())) {
    bsq[idx] = (TB_squares)sq.as_idx();
    bpc[idx] = tb_ROOK;
    idx++;
  }
  for (auto sq : (board.theirs() & board.bishops())) {
    bsq[idx] = (TB_squares)sq.as_idx();
    bpc[idx] = tb_BISHOP;
    idx++;
  }
  for (auto sq : (board.theirs() & board.pawns())) {
    bsq[idx] = (TB_squares)sq.as_idx();
    bpc[idx] = tb_PAWN;
    idx++;
  }
  bsq[idx] = tb_NOSQUARE;
  bpc[idx] = tb_NOPIECE;

  tb_probe_hard(stm, epsq, tb_NOCASTLE, wsq, bsq, wpc, bpc, &info, &dtm);
}

void ChangeInputFormat(int newInputFormat, V7TrainingData* data,
                       const PositionHistory& history) {
  data->input_format = newInputFormat;
  auto input_format =
      static_cast<pblczero::NetworkFormat::InputFormat>(newInputFormat);

  // Populate planes.
  int transform;
  InputPlanes planes = EncodePositionForNN(input_format, history, 8,
                                           FillEmptyHistory::NO, &transform);
  int plane_idx = 0;
  for (auto& plane : data->planes) {
    plane = ReverseBitsInBytes(planes[plane_idx++].mask);
  }

  // Check if current transform matches the one stored in invariance_info (bits 0-2)
  if ((data->invariance_info & 7) != transform) {
    // Probabilities need reshuffling.
    float newProbs[1858];
    std::fill(std::begin(newProbs), std::end(newProbs), -1.0f);
    bool played_fixed = false;
    bool best_fixed = false;
    bool opp_played_fixed = false; // For V7
    bool next_played_fixed = false; // For V7

    auto legal_moves = history.Last().GetBoard().GenerateLegalMoves();

    for (auto move : legal_moves) {
      int current_nn_idx = MoveToNNIndex(move, transform); // Index for new transform
      int old_nn_idx = MoveToNNIndex(move, data->invariance_info & 7); // Index for old transform
      
      if (old_nn_idx >=0 && old_nn_idx < 1858) { // Ensure old index is valid before accessing probs
          newProbs[current_nn_idx] = data->probabilities[old_nn_idx];
      } else {
          // This case should ideally not happen if old_nn_idx was valid for a legal move
          // For safety, if it does, mark newProbs as illegal (-1), or handle error
          newProbs[current_nn_idx] = -1.0f; 
      }

      // For V6+ data only (visits > 0 implies it's not an ancient upgraded format), update indices.
      if (data->visits > 0) {
        if (data->played_idx == old_nn_idx && !played_fixed) {
          data->played_idx = current_nn_idx;
          played_fixed = true;
        }
        if (data->best_idx == old_nn_idx && !best_fixed) {
          data->best_idx = current_nn_idx;
          best_fixed = true;
        }
        // V7 specific indices
        if (data->opp_played_idx == old_nn_idx && !opp_played_fixed && data->opp_played_idx != 0xFFFF) {
            data->opp_played_idx = current_nn_idx;
            opp_played_fixed = true;
        }
        if (data->next_played_idx == old_nn_idx && !next_played_fixed && data->next_played_idx != 0xFFFF) {
            data->next_played_idx = current_nn_idx;
            next_played_fixed = true;
        }
      }
    }
    // Handle sentinel values for V7 indices if they were not remapped
    if (data->visits > 0) {
        if (data->opp_played_idx != 0xFFFF && !opp_played_fixed) data->opp_played_idx = 0xFFFF; // Mark as invalid if not found
        if (data->next_played_idx != 0xFFFF && !next_played_fixed) data->next_played_idx = 0xFFFF; // Mark as invalid if not found
    }

    for (int i = 0; i < 1858; i++) {
      data->probabilities[i] = newProbs[i];
    }
  }


  const auto& position = history.Last();
  const auto& castlings = position.GetBoard().castlings();
  // Populate castlings.
  // For non-frc trained nets, just send 1 like we used to.
  uint8_t our_queen_side = 1;
  uint8_t our_king_side = 1;
  uint8_t their_queen_side = 1;
  uint8_t their_king_side = 1;
  // If frc trained, send the bit mask representing rook position.
  if (Is960CastlingFormat(input_format)) {
    our_queen_side <<= castlings.our_queenside_rook.idx;
    our_king_side <<= castlings.our_kingside_rook.idx;
    their_queen_side <<= castlings.their_queenside_rook.idx;
    their_king_side <<= castlings.their_kingside_rook.idx;
  }

  data->castling_us_ooo = castlings.we_can_000() ? our_queen_side : 0;
  data->castling_us_oo = castlings.we_can_00() ? our_king_side : 0;
  data->castling_them_ooo = castlings.they_can_000() ? their_queen_side : 0;
  data->castling_them_oo = castlings.they_can_00() ? their_king_side : 0;

  // Save the bits that aren't connected to the input_format (bits 3-7 of invariance_info).
  uint8_t invariance_non_transform_bits = data->invariance_info & 0xF8; // Mask for bits 3 through 7
  // Other params.
  if (IsCanonicalFormat(input_format)) {
    data->side_to_move_or_enpassant =
        position.GetBoard().en_passant().as_int() >> 56;
    if ((transform & FlipTransform) != 0) {
      data->side_to_move_or_enpassant =
          ReverseBitsInBytes(data->side_to_move_or_enpassant);
    }
    // Send transform (bits 0-2) and side to move (bit 7).
    data->invariance_info =
        transform | (position.IsBlackToMove() ? (1u << 7) : 0u);
  } else {
    data->side_to_move_or_enpassant = position.IsBlackToMove() ? 1 : 0;
    data->invariance_info = 0; // Non-canonical doesn't store transform bits here.
  }
  // Restore the non-transform bits.
  data->invariance_info |= invariance_non_transform_bits;
}

int ResultForData(const V7TrainingData& data) {
  // Ensure we aren't reprocessing some data that has had custom adjustments to
  // result training target applied.
  DataAssert(data.result_q == -1.0f || data.result_q == 1.0f ||
             data.result_q == 0.0f);
  // Paranoia - ensure int cast never breaks the value.
  DataAssert(data.result_q ==
             static_cast<float>(static_cast<int>(data.result_q)));
  return static_cast<int>(data.result_q);
}

std::string AsNnueString(const Position& p, Move m, float q, int result) {
  std::ostringstream out;
  out << "fen " << GetFen(p) << std::endl;
  if (p.IsBlackToMove()) m.Flip();
  out << "move " << m.ToString(false) << std::endl;
  // Formula from PR1477 adjusted for SF PawnValueEg.
  out << "score " << round(660.6 * q / (1 - 0.9751875 * std::pow(q, 10)))
      << std::endl;
  out << "ply " << p.GetGamePly() << std::endl;
  out << "result " << result << std::endl;
  out << "e" << std::endl;
  return out.str();
}

struct ProcessFileFlags {
  bool delete_files : 1;
  bool nnue_best_score : 1;
  bool nnue_best_move : 1;
};

void ProcessFile(const std::string& file, SyzygyTablebase* tablebase,
                 std::string outputDir, float distTemp, float distOffset,
                 float dtzBoost, int newInputFormat,
                 std::string nnue_plain_file, ProcessFileFlags flags) {
  // Scope to ensure reader and writer are closed before deleting source file.
  {
    try {
      TrainingDataReader reader(file);
      std::vector<V7TrainingData> fileContents;
      V7TrainingData data;
      while (reader.ReadChunk(&data)) {
        fileContents.push_back(data);
      }
      Validate(fileContents);
      MoveList moves;
      for (size_t i = 1; i < fileContents.size(); i++) {
        moves.push_back(
            DecodeMoveFromInput(PlanesFromTrainingData(fileContents[i]), // Current pos
                                PlanesFromTrainingData(fileContents[i - 1])) // Previous pos
                                );
        // All moves decoded are from the point of view of the side after the
        // move so need to mirror them all to be applicable to apply to the
        // position before.
        moves.back().Flip();
      }
      Validate(fileContents, moves);
      games += 1;
      positions += fileContents.size();
      PositionHistory history;
      int rule50ply;
      int gameply;
      ChessBoard board;
      auto input_format = static_cast<pblczero::NetworkFormat::InputFormat>(
          fileContents[0].input_format);
      PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                    &board, &rule50ply, &gameply);
      history.Reset(board, rule50ply, gameply);
      uint64_t rootHash = HashCat(board.Hash(), rule50ply);
      if (policy_subs.count(rootHash)) { // Use .count for C++11 map::find equivalent check
        PolicySubNode* rootNode = &policy_subs[rootHash];
        PositionHistory sub_history_copy = history; // Create a copy for this loop
        for (size_t i = 0; i < fileContents.size(); i++) {
          if (rootNode->active) {
            /* Some logic for choosing a softmax to apply to better align the
            new policy with the old policy...
            double bestkld =
              std::numeric_limits<double>::max(); float besttemp = 1.0f;
            // Minima is usually in this range for 'better' data.
            for (float temp = 1.0f; temp < 3.0f; temp += 0.1f) {
              float soft[1858];
              float sum = 0.0f;
              for (int j = 0; j < 1858; j++) {
                if (rootNode->policy[j] >= 0.0) {
                  soft[j] = std::pow(rootNode->policy[j], 1.0f / temp);
                  sum += soft[j];
                } else {
                  soft[j] = -1.0f;
                }
              }
              double kld = 0.0;
              for (int j = 0; j < 1858; j++) {
                if (soft[j] >= 0.0) soft[j] /= sum;
                if (rootNode->policy[j] > 0.0 &&
                    fileContents[i].probabilities[j] > 0) {
                  kld += -1.0f * soft[j] *
                    std::log(fileContents[i].probabilities[j] / soft[j]);
                }
              }
              if (kld < bestkld) {
                bestkld = kld;
                besttemp = temp;
              }
            }
            std::cerr << i << " " << besttemp << " " << bestkld << std::endl;
            */
            for (int j = 0; j < 1858; j++) {
              /*
              if (rootNode->policy[j] >= 0.0) {
                std::cerr << i << " " << j << " " << rootNode->policy[j] << " "
                          << fileContents[i].probabilities[j] << std::endl;
              }
              */
              fileContents[i].probabilities[j] = rootNode->policy[j];
            }
          }
          if (i < moves.size()) { // Check bounds for moves access
            // Determine transform for current position in sub_history_copy for MoveToNNIndex
            int current_transform = TransformForPosition(static_cast<pblczero::NetworkFormat::InputFormat>(fileContents[i].input_format), sub_history_copy);
            int idx = MoveToNNIndex(moves[i], current_transform);
            if (rootNode->children[idx] == nullptr) {
              break;
            }
            rootNode = rootNode->children[idx];
            sub_history_copy.Append(moves[i]); // Advance the sub-history
          }
        }
      }

      // Reset history for main processing pass
      PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                    &board, &rule50ply, &gameply);
      history.Reset(board, rule50ply, gameply);
      int last_rescore = -1;
      orig_counts[ResultForData(fileContents[0]) + 1]++;
      fixed_counts[ResultForData(fileContents[0]) + 1]++;
      for (int i = 0; i < static_cast<int>(moves.size()); i++) {
        // Append happens *before* processing the state *after* the move
        const auto& pre_move_board = history.Last().GetBoard(); // Board before move
        history.Append(moves[i]); // Board after move
        const auto& current_board_state = history.Last(); // Position state after moves[i]
        
        if (current_board_state.GetBoard().castlings().no_legal_castle() &&
            current_board_state.GetRule50Ply() == 0 &&
            (current_board_state.GetBoard().ours() | current_board_state.GetBoard().theirs()).count() <=
                tablebase->max_cardinality()) {
          ProbeState state;
          WDLScore wdl = tablebase->probe_wdl(current_board_state, &state);
          // Only fail state means the WDL is wrong, probe_wdl may produce
          // correct result with a stat other than OK.
          if (state != FAIL) {
            int8_t score_to_apply = 0;
            if (wdl == WDL_WIN) {
              score_to_apply = 1;
            } else if (wdl == WDL_LOSS) {
              score_to_apply = -1;
            }
            // Rescore from current chunk (i+1) back to last_rescore+1
            // Score to apply is for side to move in fileContents[j]
            // If fileContents[j] is white's move, score_to_apply is from white's POV
            // If fileContents[j] is black's move, score_to_apply is from black's POV
            // The wdl is from current_board_state's POV.
            // If current_board_state is black to move, wdl is black's score.
            // If current_board_state is white to move, wdl is white's score.
            // We need to adjust score_to_apply based on whose turn it is in fileContents[j]
            
            // `score_to_apply` is from the PoV of the player whose turn it is in `current_board_state` (i.e., `history.Last()`)
            // `fileContents[j]` is the state *before* move `j` was made.
            // `fileContents[i+1]` is the state `current_board_state`.
            // The loop goes from `j = i+1` down to `last_rescore + 1`.
            // `score_to_apply_for_chunk_j` needs to be relative to `fileContents[j]`'s side to move.

            int8_t score_for_current_chunk = score_to_apply;

            for (int j = i + 1; j > last_rescore; j--) {
              // Determine if fileContents[j] was black to move
              bool chunk_j_is_black_to_move = fileContents[j].side_to_move_or_enpassant;
              if (IsCanonicalFormat(static_cast<pblczero::NetworkFormat::InputFormat>(fileContents[j].input_format))) {
                  chunk_j_is_black_to_move = (fileContents[j].invariance_info & (1u << 7)) != 0;
              }

              // Determine if current_board_state (which generated score_for_current_chunk) was black to move
              bool current_state_is_black_to_move = current_board_state.IsBlackToMove();
              
              int8_t final_score_to_apply_for_j = score_for_current_chunk;
              if (chunk_j_is_black_to_move != current_state_is_black_to_move) {
                  // If PoVs are different (e.g. chunk is white's move, current state is black's move after white's move)
                  // then the score needs to be flipped for the chunk.
                  // This only happens if an odd number of moves occurred between chunk j and current_board_state.
                  // Since score_for_current_chunk is updated by score_for_current_chunk = -score_for_current_chunk, this is handled.
              }


              if (ResultForData(fileContents[j]) != final_score_to_apply_for_j) {
                if (j == i + 1 && last_rescore == -1) { // First rescore of the game
                  fixed_counts[ResultForData(fileContents[0]) + 1]--;
                  // The game outcome is determined by the rescore of the first relevant position.
                  // If fileContents[0] is white to move, fixed_counts should reflect white's outcome.
                  // final_score_to_apply_for_j is from PoV of fileContents[j]'s side to move.
                  // Need to adjust if fileContents[0]'s side to move is different.
                  bool game_start_is_black_to_move = fileContents[0].side_to_move_or_enpassant;
                   if (IsCanonicalFormat(static_cast<pblczero::NetworkFormat::InputFormat>(fileContents[0].input_format))) {
                       game_start_is_black_to_move = (fileContents[0].invariance_info & (1u << 7)) != 0;
                   }
                  int8_t game_outcome_score = final_score_to_apply_for_j;
                  if (chunk_j_is_black_to_move != game_start_is_black_to_move) {
                      game_outcome_score = -game_outcome_score;
                  }
                  fixed_counts[game_outcome_score + 1]++;
                }
                rescored += 1;
                delta += abs(ResultForData(fileContents[j]) - final_score_to_apply_for_j);
              }

              if (final_score_to_apply_for_j == 0) {
                fileContents[j].result_d = 1.0f;
                fileContents[j].result_q = 0.0f;
              } else {
                fileContents[j].result_d = 0.0f;
                fileContents[j].result_q = static_cast<float>(final_score_to_apply_for_j);
              }
              score_for_current_chunk = -score_for_current_chunk; // Flip for previous ply
            }
            last_rescore = i + 1;
          }
        }
      }
      PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                    &board, &rule50ply, &gameply);
      history.Reset(board, rule50ply, gameply);
      for (size_t i = 0; i < moves.size(); i++) {
        history.Append(moves[i]);
        const auto& current_board_state = history.Last(); // Position state after moves[i]
        if (current_board_state.GetBoard().castlings().no_legal_castle() &&
            current_board_state.GetRule50Ply() != 0 && // Rule 50 count is non-zero
            (current_board_state.GetBoard().ours() | current_board_state.GetBoard().theirs()).count() <=
                tablebase->max_cardinality()) {
          ProbeState state;
          WDLScore wdl = tablebase->probe_wdl(current_board_state, &state);
          if (state != FAIL) {
            int8_t tb_score = 0; // Score from TB from PoV of current_board_state
            if (wdl == WDL_WIN) {
              tb_score = 1;
            } else if (wdl == WDL_LOSS) {
              tb_score = -1;
            }
            
            // fileContents[i+1] corresponds to current_board_state
            int8_t game_result_for_chunk = ResultForData(fileContents[i+1]);
            int8_t new_score_for_chunk = game_result_for_chunk; // Default to original result

            if (game_result_for_chunk != tb_score) {
                new_score_for_chunk = 0; // Disagreement means draw
            }

            bool dtz_rescored = false;
            if (game_result_for_chunk != tb_score && tb_score != 0) {
              int steps = current_board_state.GetRule50Ply();
              bool no_reps = true;
              // Check reps in history leading up to current_board_state relevant to rule50 count
              PositionHistory temp_hist_for_reps = history; // Copy current history
              for (int k = 0; k < steps; k++) {
                  if (temp_hist_for_reps.GetLength() <= 1) {no_reps = false; break;} // Not enough history
                  temp_hist_for_reps.Pop(); // Go back one ply
                  if (temp_hist_for_reps.Last().GetRepetitions() !=0) {
                      no_reps = false;
                      break;
                  }
              }

              if (no_reps) {
                int depth = tablebase->probe_dtz(current_board_state, &state);
                if (state != FAIL) {
                  if (steps + std::abs(depth) < 99) { // Check against 50-move rule (100 plies)
                    rescored3++;
                    new_score_for_chunk = tb_score; // DTZ confirms TB score
                    dtz_rescored = true;
                  }
                }
              }
            }

            if (new_score_for_chunk != 0 && tb_score != 0 && !dtz_rescored) {
              int depth = tablebase->probe_dtz(current_board_state, &state);
              if (state != FAIL) {
                int steps = current_board_state.GetRule50Ply();
                if (steps + std::abs(depth) > 101) { // Generous threshold for forcing draw
                  rescored3++;
                  new_score_for_chunk = 0; // DTZ indicates draw by 50-move
                  dtz_rescored = true;
                }
              }
            }
            if (new_score_for_chunk != game_result_for_chunk) {
              rescored2 += 1;
            }

            if (new_score_for_chunk == 0) {
              fileContents[i + 1].result_d = 1.0f;
              fileContents[i+1].result_q = 0.0f;
            } else {
              fileContents[i + 1].result_d = 0.0f;
              fileContents[i+1].result_q = static_cast<float>(new_score_for_chunk);
            }
          }
        }
      }

      if (distTemp != 1.0f || distOffset != 0.0f || dtzBoost != 0.0f) {
        PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                      &board, &rule50ply, &gameply);
        history.Reset(board, rule50ply, gameply);
        for (size_t chunk_idx = 0; chunk_idx < fileContents.size(); ++chunk_idx) {
          auto& chunk = fileContents[chunk_idx];
          const auto& current_pos_for_chunk = history.Last(); // Position for this chunk
          std::vector<bool> boost_probs(1858, false);
          int boost_count = 0;

          if (dtzBoost != 0.0f && current_pos_for_chunk.GetBoard().castlings().no_legal_castle() &&
              (current_pos_for_chunk.GetBoard().ours() | current_pos_for_chunk.GetBoard().theirs()).count() <=
                  tablebase->max_cardinality()) {
            MoveList to_boost;
            MoveList dtz_moves_rule50_aware; // Moves from root_probe considering 50-move rule
            MoveList dtz_moves_no_rule50;    // Moves from root_probe ignoring 50-move rule (for Gaviota check)

            tablebase->root_probe(current_pos_for_chunk, true, true, &dtz_moves_rule50_aware); // true for rule50
            
            // If current position has repetitions since last zeroing move, use rule50_aware moves for boosting
            // Otherwise, use moves ignoring rule50 for potential Gaviota DTM check.
            if (current_pos_for_chunk.DidRepeatSinceLastZeroingMove()) {
                to_boost = dtz_moves_rule50_aware;
            } else {
                tablebase->root_probe(current_pos_for_chunk, false, true, &dtz_moves_no_rule50); // false for rule50
                to_boost = dtz_moves_no_rule50; // Start with these, might be refined by Gaviota
            }


            if (gaviotaEnabled && to_boost.size() > 1 &&
                (current_pos_for_chunk.GetBoard().ours() | current_pos_for_chunk.GetBoard().theirs()).count() <= 5) {
              std::vector<unsigned int> dtms;
              unsigned int mininum_dtm = 1000; // Large value
              
              for (const auto& move : to_boost) {
                Position next_pos = Position(current_pos_for_chunk, move);
                unsigned int info;
                unsigned int dtm_val;
                gaviota_tb_probe_hard(next_pos, info, dtm_val);
                dtms.push_back(dtm_val);
                if (dtm_val < mininum_dtm) mininum_dtm = dtm_val;
              }

              if (mininum_dtm < 1000) { // If a valid DTM was found
                MoveList gaviota_boost_moves;
                for (size_t k=0; k < to_boost.size(); ++k) {
                    if (dtms[k] == mininum_dtm) {
                        gaviota_boost_moves.push_back(to_boost[k]);
                    }
                }
                if (!gaviota_boost_moves.empty()) {
                    to_boost = gaviota_boost_moves; // Refine to_boost with Gaviota results
                    policy_dtm_bump++;
                }
              }
            }
            // If after Gaviota check to_boost is empty but dtz_moves_rule50_aware was not, revert.
            // This handles cases where Gaviota doesn't find a DTM or to_boost became empty.
            if (to_boost.empty() && !dtz_moves_rule50_aware.empty() && current_pos_for_chunk.DidRepeatSinceLastZeroingMove()){
                to_boost = dtz_moves_rule50_aware;
            }


            int transform = TransformForPosition(static_cast<pblczero::NetworkFormat::InputFormat>(chunk.input_format), current_pos_for_chunk);
            for (auto& move : to_boost) {
              boost_probs[MoveToNNIndex(move, transform)] = true;
            }
            boost_count = to_boost.size();
          }
          float sum = 0.0;
          int prob_index = 0;
          float preboost_sum = 0.0f;
          for (auto& prob : chunk.probabilities) {
            float offset =
                distOffset +
                (boost_probs[prob_index] && boost_count > 0 ? (dtzBoost / boost_count) : 0.0f);
            if (dtzBoost != 0.0f && boost_probs[prob_index]) {
              preboost_sum += prob;
              if (prob < 0 || std::isnan(prob))
                std::cerr << "Bump for move that is illegal????" << std::endl;
              policy_bump++;
            }
            prob_index++;
            if (prob < 0 || std::isnan(prob)) continue;
            prob = std::max(0.0f, prob + offset);
            prob = std::pow(prob, 1.0f / distTemp);
            sum += prob;
          }
          prob_index = 0;
          float boost_sum = 0.0f;
          for (auto& prob : chunk.probabilities) {
            if (prob < 0 || std::isnan(prob)) { // Check before division
                prob_index++; // Increment even for illegal moves
                continue;
            }
            if (sum > 0) prob /= sum; // Avoid division by zero
            else prob = 0; // Or handle as error / set to uniform if sum is 0 and there are legal moves

            if (dtzBoost != 0.0f && boost_probs[prob_index]) {
              boost_sum += prob;
            }
            prob_index++;
          }
          if (boost_count > 0) {
            policy_nobump_total_hist[static_cast<int>(std::min(10.0f, preboost_sum * 10.0f))]++;
            policy_bump_total_hist[static_cast<int>(std::min(10.0f, boost_sum * 10.0f))]++;
          }
          
          if (chunk_idx < moves.size()) { // Advance history for next iteration
            history.Append(moves[chunk_idx]);
          }
        }
      }

      // Make move_count field plies_left for moves left head.
      int offset = 0;
      bool all_draws = true;
      for (auto& chunk : fileContents) {
        // plies_left can't be 0 for real v5 data (which is upgraded to v6/v7),
        // so if it is 0 it must be a v3/v4 conversion.
        if (chunk.plies_left == 0.0f && chunk.version < 5) { // Check original version if available, or rely on reader upgrade path
          chunk.plies_left = static_cast<float>(fileContents.size() - offset);
        }
        offset++;
        all_draws = all_draws && (ResultForData(chunk) == 0);
      }

      // Correct plies_left using Gaviota TBs for 5 piece and less positions.
      if (gaviotaEnabled && !all_draws) {
        PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                      &board, &rule50ply, &gameply);
        history.Reset(board, rule50ply, gameply);
        int last_gaviota_rescore_ply = -1; // Track the ply index of the last position whose plies_left was set by Gaviota
        for (size_t i = 0; i < fileContents.size(); i++) { // Iterate through all chunks
            const auto& current_chunk = fileContents[i];
            const auto& current_pos_state = history.Last(); // Position corresponding to current_chunk

            if ((ResultForData(current_chunk) != 0) &&
              current_pos_state.GetBoard().castlings().no_legal_castle() &&
              (current_pos_state.GetBoard().ours() | current_pos_state.GetBoard().theirs()).count() <= 5) {
            
              unsigned int info;
              unsigned int dtm;
              gaviota_tb_probe_hard(current_pos_state, info, dtm);

              if (info == tb_WMATE || info == tb_BMATE) { // Is a mate for one player
                int steps = current_pos_state.GetRule50Ply();
                if (!((dtm + steps > 99) && (dtm <= current_chunk.plies_left))) { // Condition from python
                    // Check for repetitions since last zeroing move
                    bool no_reps_since_zeroing = true;
                    PositionHistory temp_hist_for_reps = history;
                    for(int k=0; k < steps; ++k) {
                        if(temp_hist_for_reps.GetLength() <=1) {no_reps_since_zeroing=false; break;}
                        temp_hist_for_reps.Pop();
                        if(temp_hist_for_reps.Last().GetRepetitions() !=0) {
                            no_reps_since_zeroing=false;
                            break;
                        }
                    }

                    if (no_reps_since_zeroing) {
                        gaviota_dtm_rescores++;
                        // Rescore plies_left for current_chunk and preceding chunks back to last_gaviota_rescore_ply + 1
                        for (int j = static_cast<int>(i); j > last_gaviota_rescore_ply; --j) {
                            fileContents[j].plies_left = static_cast<float>(dtm + (i - j));
                        }
                        last_gaviota_rescore_ply = static_cast<int>(i);
                    }
                }
              }
            }
            if (i < moves.size()) { // Append move to advance history for next chunk
                history.Append(moves[i]);
            }
        }
      }


      // Correct plies_left using DTZ for 3 piece no-pawn positions only.
      // If Gaviota TBs are enabled no need to use syzygy for this.
      if (!gaviotaEnabled && !all_draws) {
        PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                      &board, &rule50ply, &gameply);
        history.Reset(board, rule50ply, gameply);
        for (size_t i = 0; i < fileContents.size(); i++) { // Iterate all chunks
          const auto& current_pos_state = history.Last(); // Position for fileContents[i]

          if (current_pos_state.GetBoard().castlings().no_legal_castle() &&
              (current_pos_state.GetBoard().ours() | current_pos_state.GetBoard().theirs()).count() <= 3 &&
              current_pos_state.GetBoard().pawns().empty()) {
            ProbeState state;
            WDLScore wdl = tablebase->probe_wdl(current_pos_state, &state);
            if (state != FAIL) {
              int8_t tb_score = 0;
              if (wdl == WDL_WIN) tb_score = 1;
              else if (wdl == WDL_LOSS) tb_score = -1;

              if (tb_score != 0) { // No point updating for draws
                int steps = current_pos_state.GetRule50Ply();
                bool no_reps_since_zeroing = true;
                PositionHistory temp_hist_for_reps = history;
                for(int k=0; k < steps; ++k) {
                    if(temp_hist_for_reps.GetLength() <=1) {no_reps_since_zeroing=false; break;}
                    temp_hist_for_reps.Pop();
                    if(temp_hist_for_reps.Last().GetRepetitions() !=0) {
                        no_reps_since_zeroing=false;
                        break;
                    }
                }

                if (no_reps_since_zeroing) {
                  int dtz_depth = tablebase->probe_dtz(current_pos_state, &state);
                  if (state != FAIL) {
                    int converted_ply_remaining = std::abs(dtz_depth);
                    if (steps + std::abs(dtz_depth) < 99) {
                      fileContents[i].plies_left = static_cast<float>(converted_ply_remaining);
                      // If rule50 is 0, propagate this dtz back
                      if (steps == 0) {
                          for (int j = static_cast<int>(i) - 1; j >=0; --j) {
                              fileContents[j].plies_left = static_cast<float>(converted_ply_remaining + (i-j));
                          }
                      }
                    }
                  }
                }
              }
            }
          }
          if (i < moves.size()) { // Advance history
              history.Append(moves[i]);
          }
        }
      }
      // Deblunder only works from v6 data onwards. We therefore check
      // the visits field which is 0 if we're dealing with upgraded data from ancient formats.
      if (deblunderEnabled && fileContents.back().visits > 0) {
        // Reconstruct history up to the point before Syzygy cut-off if any, or end of game.
        PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                      &board, &rule50ply, &gameply);
        history.Reset(board, rule50ply, gameply);
        size_t deblunder_hist_len = 0;
        for (size_t i = 0; i < moves.size(); ++i) {
            history.Append(moves[i]);
            deblunder_hist_len = i + 1;
            const auto& current_board = history.Last().GetBoard();
             if (current_board.castlings().no_legal_castle() &&
                (current_board.ours() | current_board.theirs()).count() <= tablebase->max_cardinality()) {
                // history.Pop(); // No, keep history at this point. Deblunder from here backwards.
                break; 
            }
        }


        float activeZ_q = fileContents[deblunder_hist_len].result_q;
        float activeZ_d = fileContents[deblunder_hist_len].result_d;
        float activeZ_m = fileContents[deblunder_hist_len].plies_left;
        bool deblunderingStarted = false;

        for (int k = static_cast<int>(deblunder_hist_len); k >=0; --k) { // Iterate backwards from deblunder_hist_len (chunk index)
          auto& cur_chunk = fileContents[k];
          bool deblunderTriggerThreshold =
              (cur_chunk.best_q - cur_chunk.played_q >
               deblunderQBlunderThreshold - deblunderQBlunderWidth / 2.0);
          bool deblunderTriggerTerminal =
              (cur_chunk.best_q > -1.0f && cur_chunk.played_q < 1.0f && // best_q not a loss, played_q not a win
               ((cur_chunk.best_q == 1.0f && ((cur_chunk.invariance_info & 8) != 0)) || // best move is proven win
                cur_chunk.played_q == -1.0f)); // played move is a loss

          if (deblunderTriggerThreshold || deblunderTriggerTerminal) {
            float newZRatio = 1.0f;
            if (deblunderQBlunderWidth > 0 && !deblunderTriggerTerminal) {
              newZRatio = std::min(1.0f, (cur_chunk.best_q - cur_chunk.played_q -
                                          (deblunderQBlunderThreshold - deblunderQBlunderWidth/2.0f)) / // Adjusted threshold start
                                                 deblunderQBlunderWidth);
              newZRatio = std::max(0.0f, newZRatio); // Clamp to [0,1]
            }
            activeZ_q = (1.0f - newZRatio) * activeZ_q + newZRatio * cur_chunk.best_q;
            activeZ_d = (1.0f - newZRatio) * activeZ_d + newZRatio * cur_chunk.best_d;
            activeZ_m = (1.0f - newZRatio) * activeZ_m + newZRatio * cur_chunk.best_m;
            deblunderingStarted = true;
            blunders += 1;
          }
          if (deblunderingStarted) {
            cur_chunk.result_q = activeZ_q;
            cur_chunk.result_d = activeZ_d;
            cur_chunk.plies_left = activeZ_m;
          }
          
          if (k == 0) break; // Stop if we processed the first chunk

          // For next iteration (previous ply)
          activeZ_q = -activeZ_q; // Flip Q for opponent
          activeZ_m += 1.0f;      // Increment plies
          // history.Pop(); // Not needed as we are just iterating chunks
        }
      }
      if (newInputFormat != -1) {
        PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                      &board, &rule50ply, &gameply);
        history.Reset(board, rule50ply, gameply);
        ChangeInputFormat(newInputFormat, &fileContents[0], history);
        for (size_t i = 0; i < moves.size(); i++) {
          history.Append(moves[i]);
          ChangeInputFormat(newInputFormat, &fileContents[i + 1], history);
        }
      }

      if (!outputDir.empty()) {
        std::string fileName = file.substr(file.find_last_of("/\\") + 1);
        TrainingDataWriter writer(outputDir + "/" + fileName);
        for (auto& chunk : fileContents) { // Use reference to avoid copy
          // Don't save chunks that just provide move history.
          // Bit 6 (value 64) of invariance_info marks deletion.
          if ((chunk.invariance_info & (1u << 6)) == 0) {
            writer.WriteChunk(chunk);
          }
        }
      }

      // Output data in Stockfish plain format.
      if (!nnue_plain_file.empty()) {
        static Mutex mutex;
        std::ostringstream out;
        pblczero::NetworkFormat::InputFormat current_file_input_format;
        if (newInputFormat != -1) {
          current_file_input_format =
              static_cast<pblczero::NetworkFormat::InputFormat>(newInputFormat);
        } else {
          current_file_input_format = input_format; // Original input format of the file
        }
        PopulateBoard(current_file_input_format, PlanesFromTrainingData(fileContents[0]), &board,
                      &rule50ply, &gameply);
        history.Reset(board, rule50ply, gameply);
        for (size_t i = 0; i < fileContents.size(); i++) {
          auto& chunk = fileContents[i]; // Use reference
          Position p = history.Last();
          if (chunk.visits > 0) { // Indicates it's a full data chunk, not just placeholder
            // Determine transform for current position p
            int transform_for_p = TransformForPosition(current_file_input_format, history);
            Move m = MoveFromNNIndex(
                flags.nnue_best_move ? chunk.best_idx : chunk.played_idx,
                transform_for_p);
            float q_val = flags.nnue_best_score ? chunk.best_q : chunk.played_q;
            out << AsNnueString(p, m, q_val, round(chunk.result_q));
          } else if (i < moves.size()) { // Likely an older format chunk without visits, use move if available
             // This branch might be less relevant if reader upgrades all to have some visit count or marks them.
             // For safety, ensure there is a move to use.
            out << AsNnueString(p, moves[i], chunk.best_q, // Or some other q? best_q might be only valid one.
                                round(chunk.result_q));
          }
          if (i < moves.size()) {
            history.Append(moves[i]);
          }
        }
        std::ofstream outfile_stream; // Changed from file to outfile_stream to avoid conflict
        Mutex::Lock lock(mutex);
        outfile_stream.open(nnue_plain_file, std::ios_base::app);
        if (outfile_stream.is_open()) {
          outfile_stream << out.str();
          outfile_stream.close();
        }
      }
    } catch (Exception& ex) {
      std::cerr << "While processing: " << file
                << " - Exception thrown: " << ex.what() << std::endl;
      if (flags.delete_files) {
        std::cerr << "It will be deleted." << std::endl;
      }
    }
  }
  if (flags.delete_files) {
    remove(file.c_str());
  }
}

void ProcessFiles(const std::vector<std::string>& files,
                  SyzygyTablebase* tablebase, std::string outputDir,
                  float distTemp, float distOffset, float dtzBoost,
                  int newInputFormat, int offset, int mod,
                  std::string nnue_plain_file, ProcessFileFlags flags) {
  std::cerr << "Thread: " << offset << " starting" << std::endl;
  for (size_t i = offset; i < files.size(); i += mod) {
    if (files[i].rfind(".gz") != files[i].size() - 3) {
      std::cerr << "Skipping: " << files[i] << std::endl;
      continue;
    }
    ProcessFile(files[i], tablebase, outputDir, distTemp, distOffset, dtzBoost,
                newInputFormat, nnue_plain_file, flags);
  }
}

void BuildSubs(const std::vector<std::string>& files) {
  for (auto& file : files) {
    TrainingDataReader reader(file);
    std::vector<V7TrainingData> fileContents;
    V7TrainingData data;
    while (reader.ReadChunk(&data)) {
      fileContents.push_back(data);
    }
    Validate(fileContents);
    MoveList moves;
    for (size_t i = 1; i < fileContents.size(); i++) {
      moves.push_back(
          DecodeMoveFromInput(PlanesFromTrainingData(fileContents[i]),
                              PlanesFromTrainingData(fileContents[i - 1])));
      moves.back().Flip();
    }
    Validate(fileContents, moves);

    PositionHistory history;
    int rule50ply;
    int gameply;
    ChessBoard board;
    auto input_format = static_cast<pblczero::NetworkFormat::InputFormat>(
        fileContents[0].input_format);
    PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]), &board,
                  &rule50ply, &gameply);
    history.Reset(board, rule50ply, gameply);
    uint64_t rootHash = HashCat(board.Hash(), rule50ply);
    PolicySubNode* rootNode = &policy_subs[rootHash]; // Creates if not exists
    for (size_t i = 0; i < fileContents.size(); i++) {
      if ((fileContents[i].invariance_info & (1u << 6)) == 0) { // Not marked for deletion
        rootNode->active = true;
        for (int j = 0; j < 1858; j++) {
          rootNode->policy[j] = fileContents[i].probabilities[j];
        }
      }
      if (i < moves.size()) { // Check bounds for moves access
        int transform = TransformForPosition(input_format, history);
        int idx = MoveToNNIndex(moves[i], transform);
        if (rootNode->children[idx] == nullptr) {
          rootNode->children[idx] = new PolicySubNode();
        }
        rootNode = rootNode->children[idx];
        history.Append(moves[i]);
      }
    }
  }
}

}  // namespace

#ifdef _WIN32
#define SEP_CHAR ';'
#else
#define SEP_CHAR ':'
#endif

void RunRescorer() {
  OptionsParser options;
  orig_counts[0] = 0;
  orig_counts[1] = 0;
  orig_counts[2] = 0;
  fixed_counts[0] = 0;
  fixed_counts[1] = 0;
  fixed_counts[2] = 0;
  for (int i = 0; i < 11; i++) policy_bump_total_hist[i] = 0;
  for (int i = 0; i < 11; i++) policy_nobump_total_hist[i] = 0;
  options.Add<StringOption>(kSyzygyTablebaseId);
  options.Add<StringOption>(kGaviotaTablebaseId);
  options.Add<StringOption>(kInputDirId);
  options.Add<StringOption>(kOutputDirId);
  options.Add<StringOption>(kPolicySubsDirId);
  options.Add<IntOption>(kThreadsId, 1, 20) = 1;
  options.Add<FloatOption>(kTempId, 0.001, 100) = 1;
  options.Add<FloatOption>(kDistributionOffsetId, -0.999, 0) = 0;
  options.Add<FloatOption>(kMinDTZBoostId, 0, 1) = 0;
  options.Add<IntOption>(kNewInputFormatId, -1, 256) = -1; // Max value for InputFormat enum
  options.Add<BoolOption>(kDeblunder) = false;
  options.Add<FloatOption>(kDeblunderQBlunderThreshold, 0.0f, 2.0f) = 2.0f;
  options.Add<FloatOption>(kDeblunderQBlunderWidth, 0.0f, 2.0f) = 0.0f;
  options.Add<StringOption>(kNnuePlainFileId);
  options.Add<BoolOption>(kNnueBestScoreId) = true;
  options.Add<BoolOption>(kNnueBestMoveId) = false;
  options.Add<BoolOption>(kDeleteFilesId) = true;

  if (!options.ProcessAllFlags()) return;

  if (options.GetOptionsDict().IsDefault<std::string>(kOutputDirId) &&
      options.GetOptionsDict().IsDefault<std::string>(kNnuePlainFileId)) {
    std::cerr << "Must provide an output dir or NNUE plain file." << std::endl;
    return;
  }

  deblunderEnabled = options.GetOptionsDict().Get<bool>(kDeblunder);
  deblunderQBlunderThreshold =
      options.GetOptionsDict().Get<float>(kDeblunderQBlunderThreshold);
  deblunderQBlunderWidth =
      options.GetOptionsDict().Get<float>(kDeblunderQBlunderWidth);

  SyzygyTablebase tablebase;
  if (!options.GetOptionsDict().IsDefault<std::string>(kSyzygyTablebaseId)) {
      if (!tablebase.init(
              options.GetOptionsDict().Get<std::string>(kSyzygyTablebaseId)) ||
          tablebase.max_cardinality() < 3) { // Syzygy needs at least 3 pieces for meaningful probes here
        std::cerr << "FAILED TO LOAD SYZYGY OR MAX CARDINALITY TOO LOW" << std::endl;
        // Allow running without Syzygy if paths are not provided.
      }
  } else {
      std::cerr << "Syzygy paths not provided, Syzygy features will be disabled." << std::endl;
  }


  auto dtmPaths =
      options.GetOptionsDict().Get<std::string>(kGaviotaTablebaseId);
  if (!dtmPaths.empty()) { // Check if empty, not size != 0
    std::stringstream path_string_stream(dtmPaths);
    std::string path;
    auto paths = tbpaths_init();
    while (std::getline(path_string_stream, path, SEP_CHAR)) {
      paths = tbpaths_add(paths, path.c_str());
    }
    tb_init(0, tb_CP4, paths); // tb_MAX_PIECES = 0 for default, tb_Compression_None = tb_CP4
    tbcache_init(64 * 1024 * 1024, 64); // 64MB cache, 64 entries
    if (tb_availability() != 63) { // 63 means 3,4,5 piece EGTs are available
      std::cerr << "UNEXPECTED Gaviota availability. Expected 3,4,5 piece TBs (mask 63). Got: " << tb_availability() << std::endl;
      // Allow to continue, but Gaviota features might not work as expected.
    } else {
      std::cerr << "Found Gaviota TBs" << std::endl;
      gaviotaEnabled = true;
    }
  } else {
      std::cerr << "Gaviota paths not provided, Gaviota features will be disabled." << std::endl;
  }

  auto policySubsDir =
      options.GetOptionsDict().Get<std::string>(kPolicySubsDirId);
  if (!policySubsDir.empty()) { // Check if empty
    auto policySubFiles = GetFileList(policySubsDir);
    for (size_t i = 0; i < policySubFiles.size(); i++) {
      policySubFiles[i] = policySubsDir + "/" + policySubFiles[i];
    }
    BuildSubs(policySubFiles);
  }

  auto inputDir = options.GetOptionsDict().Get<std::string>(kInputDirId);
  if (inputDir.empty()) { // Check if empty
    std::cerr << "Must provide an input dir." << std::endl;
    return;
  }
  auto files = GetFileList(inputDir);
  if (files.empty()) { // Check if empty
    std::cerr << "No files to process" << std::endl;
    return;
  }
  for (size_t i = 0; i < files.size(); i++) {
    files[i] = inputDir + "/" + files[i];
  }
  float dtz_boost = options.GetOptionsDict().Get<float>(kMinDTZBoostId);
  unsigned int threads_val = options.GetOptionsDict().Get<int>(kThreadsId); // Renamed to avoid conflict
  ProcessFileFlags flags;
  flags.delete_files = options.GetOptionsDict().Get<bool>(kDeleteFilesId);
  flags.nnue_best_score = options.GetOptionsDict().Get<bool>(kNnueBestScoreId);
  flags.nnue_best_move = options.GetOptionsDict().Get<bool>(kNnueBestMoveId);
  if (threads_val > 1) {
    std::vector<std::thread> threads_vec; // Renamed
    int offset = 0;
    while (threads_vec.size() < threads_val) {
      int offset_val = offset;
      offset++;
      threads_vec.emplace_back([&options, offset_val, &files, &tablebase, threads_val, // Pass files by ref
                             dtz_boost, flags]() { // Pass tablebase by ref
        ProcessFiles(
            files, &tablebase,
            options.GetOptionsDict().Get<std::string>(kOutputDirId),
            options.GetOptionsDict().Get<float>(kTempId),
            options.GetOptionsDict().Get<float>(kDistributionOffsetId),
            dtz_boost, options.GetOptionsDict().Get<int>(kNewInputFormatId),
            offset_val, threads_val,
            options.GetOptionsDict().Get<std::string>(kNnuePlainFileId), flags);
      });
    }
    for (size_t i = 0; i < threads_vec.size(); i++) {
      threads_vec[i].join();
    }

  } else {
    ProcessFiles(
        files, &tablebase,
        options.GetOptionsDict().Get<std::string>(kOutputDirId),
        options.GetOptionsDict().Get<float>(kTempId),
        options.GetOptionsDict().Get<float>(kDistributionOffsetId), dtz_boost,
        options.GetOptionsDict().Get<int>(kNewInputFormatId), 0, 1,
        options.GetOptionsDict().Get<std::string>(kNnuePlainFileId), flags);
  }
  std::cout << "Games processed: " << games << std::endl;
  std::cout << "Positions processed: " << positions << std::endl;
  std::cout << "Rescores performed: " << rescored << std::endl;
  std::cout << "Cumulative outcome change: " << delta << std::endl;
  std::cout << "Secondary rescores performed: " << rescored2 << std::endl;
  std::cout << "Secondary rescores performed used dtz: " << rescored3
            << std::endl;
  std::cout << "Blunders picked up by deblunder threshold: " << blunders
            << std::endl;
  std::cout << "Number of policy values boosted by dtz or dtm " << policy_bump
            << std::endl;
  std::cout << "Number of policy values boosted by dtm " << policy_dtm_bump
            << std::endl;
  std::cout << "Orig policy_sum dist of boost candidate:";
  std::cout << std::endl;
  int event_sum = 0;
  for (int i = 0; i < 11; i++) event_sum += policy_bump_total_hist[i]; // Typo: policy_bump_total_hist
  if (event_sum == 0) event_sum = 1; // Avoid division by zero
  for (int i = 0; i < 11; i++) {
    std::cout << " " << std::setprecision(4)
              << (static_cast<float>(policy_nobump_total_hist[i]) / static_cast<float>(event_sum));
  }
  std::cout << std::endl;
  std::cout << "Boosted policy_sum dist of boost candidate:";
  std::cout << std::endl;
  for (int i = 0; i < 11; i++) {
    std::cout << " " << std::setprecision(4)
              << (static_cast<float>(policy_bump_total_hist[i]) / static_cast<float>(event_sum));
  }
  std::cout << std::endl;
  std::cout << "Original L: " << orig_counts[0] << " D: " << orig_counts[1]
            << " W: " << orig_counts[2] << std::endl;
  std::cout << "After L: " << fixed_counts[0] << " D: " << fixed_counts[1]
            << " W: " << fixed_counts[2] << std::endl;
  std::cout << "Gaviota DTM move_count rescores: " << gaviota_dtm_rescores
            << std::endl;
}

}  // namespace lczero
