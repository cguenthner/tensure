(ns tensure.core-test
  (:refer-clojure :exclude [min max])
  (:require [clojure.test :refer :all]
            [tensure.core :refer :all :as m]
            [tensure.test-utils :refer :all]
            [tensure.utils :as u]))

(deftest sample-uniform-test
  (testing "returns tensors of the correct shape"
    (is (scalar? (sample-uniform [])))
    (is (scalar? (sample-uniform nil)))
    (is (= (shape (sample-uniform 3)) [3]))
    (is (= (shape (sample-uniform [4])) [4]))
    (is (= (shape (sample-uniform [2 3])) [2 3]))
    (is (= (shape (sample-uniform [2 2 3])) [2 2 3])))
  (testing "returns numbers with the expected statistical properties"
    (let [nd (sample-uniform [10000] 0)
          ns (eseq nd)]
      (is (>= (apply clojure.core/min ns) 0))
      (is (< (apply clojure.core/max ns) 1))
      (is (< (Math/abs (- (scalar->number (emean nd)) 0.5)) 0.01))
      (is (< (Math/abs (- (scalar->number (estdev nd)) 0.2887)) 0.01))))
  (testing "has the correct seeding behavior"
    (is (not (equals (sample-uniform [10]) (sample-uniform [10]))))
    (is (not (equals (sample-uniform [10] 0) (sample-uniform [10] 1))))
    (is (equals (sample-uniform [10] 0) (sample-uniform [10] 0)))))

(deftest sample-normal-test
  (testing "returns tensors of the correct shape"
    (is (scalar? (sample-normal [])))
    (is (scalar? (sample-normal nil)))
    (is (= (shape (sample-normal 3)) [3]))
    (is (= (shape (sample-normal [4])) [4]))
    (is (= (shape (sample-normal [2 3])) [2 3]))
    (is (= (shape (sample-normal [2 2 3])) [2 2 3])))
  (testing "returns numbers with the expected statistical properties"
    (let [nd (sample-normal [10000] 0)]
      (is (< (Math/abs (scalar->number (emean nd))) 0.02))
      (is (< (Math/abs (- (scalar->number (estdev nd)) 1)) 0.01))))
  (testing "has the correct seeding behavior"
    (is (not (equals (sample-normal [10]) (sample-normal [10]))))
    (is (not (equals (sample-normal [10] 0) (sample-normal [10] 1))))
    (is (equals (sample-normal [10] 0) (sample-normal [10] 0)))))

(deftest sample-rand-int-test
  (testing "returns tensors of the correct shape"
    (is (scalar? (sample-rand-int [] 10)))
    (is (scalar? (sample-rand-int nil 10)))
    (is (= (shape (sample-rand-int 3 10)) [3]))
    (is (= (shape (sample-rand-int [4] 10)) [4]))
    (is (= (shape (sample-rand-int [2 3] 10)) [2 3]))
    (is (= (shape (sample-rand-int [2 2 3] 10)) [2 2 3])))
  (testing "returns numbers with the expected statistical properties"
    (let [nd-10 (eseq (sample-rand-int [10000] 10 0))
          nd-50 (eseq (sample-rand-int [10000] 50 0))]
      (is (>= (apply clojure.core/min nd-10) 0))
      (is (< (apply clojure.core/max nd-10) 10))
      (is (>= (apply clojure.core/min nd-50) 0))
      (is (< (apply clojure.core/max nd-50) 50))
      ; TODO: Check mean and SD using functions in test_utils.
      ))
  (testing "has the correct seeding behavior"
    (is (not (equals (sample-rand-int [10] 100) (sample-rand-int [10] 100))))
    (is (not (equals (sample-rand-int [10] 100 0) (sample-rand-int [10] 100 1))))
    (is (equals (sample-rand-int [10] 100 0) (sample-rand-int [10] 100 0)))))

(deftest emap-test
  (testing "returns a new tensor, not a view"
    (let [nd (array [[1 2 3] [4 5 6]])
          result (emap inc nd)]
      (is (equals result (array [[2 3 4] [5 6 7]])))
      (is (equals nd (array [[1 2 3] [4 5 6]])))))
  (testing "scalars"
    (is (equals (emap inc (array 7)) (array 8)))
    (is (equals (emap + (array 7) (array 3)) (array 10)))
    (is (thrown? Exception (emap + (array 7) (array [1 2 3])))))
  (testing "vectors"
    (is (equals (emap inc (array [1 2 3])) (array [2 3 4])))
    (is (equals (emap + (array [1 2 3]) (array [4 5 6])) (array [5 7 9])))
    (is (equals (emap #(+ (- (* 2 %1) %2) %3) (array [1 2 3]) (array [4 5 6]) (array [7 8 9])) (array [5 7 9]))))
  (testing "matrices"
    (is (equals (emap inc (array [[1 2 3] [4 5 6]])) (array [[2 3 4] [5 6 7]])))
    (is (equals (emap * (array [[1 2 3] [4 5 6]]) (array [[7 8 9] [10 11 12]]))
                (array [[7 16 27] [40 55 72]])))
    (is (thrown? Exception (emap * (array [[1 2 3] [4 5 6]]) (array [[7 8] [10 11]])))))
  (testing "tensors"
    (is (equals (emap inc (array [[[1 2] [3 4]] [[5 6] [7 8]]])) (array [[[2 3] [4 5]] [[6 7] [8 9]]])))
    (is (equals (emap + (array [[[1 2] [3 4]] [[5 6] [7 8]]]) (array [[[16 15] [14 13]] [[12 11] [10 9]]]))
                (array [[[17 17] [17 17]] [[17 17] [17 17]]])))))

(deftest emap-indexed-test
  (let [sum-indices (fn [indices _]
                      (apply + indices))
        select-val (fn [_ val] val)
        i*val (fn [indices val]
                (* (apply + indices) val))
        sum-two-vals (fn [_ a b] (+ a b))
        indices+vals (fn [indices a b] (apply + a b indices))]
    (testing "returns a new tensor, not a view"
      (let [nd (array [[1 2 3] [4 5 6]])
            result (emap-indexed (comp inc select-val) nd)]
        (is (equals result (array [[2 3 4] [5 6 7]])))
        (is (equals nd (array [[1 2 3] [4 5 6]])))))
    (testing "scalars"
      (is (equals (emap-indexed sum-indices (array 7)) (array 0)))
      (is (equals (emap-indexed select-val (array 7)) (array 7)))
      (is (equals (emap-indexed sum-two-vals (array 3) (array 7)) (array 10)))
      (is (equals (emap-indexed indices+vals (array 3) (array 7)) (array 10))))
    (testing "vectors"
      (is (equals (emap-indexed sum-indices (array [1 2 3])) (array [0 1 2])))
      (is (equals (emap-indexed i*val (array [1 2 3])) (array [0 2 6])))
      (is (equals (emap-indexed sum-two-vals (array [1 2 3]) (array [4 5 6])) (array [5 7 9])))
      (is (equals (emap-indexed indices+vals (array [1 2 3]) (array [4 5 6])) (array [5 8 11]))))
    (testing "matrices"
      (is (equals (emap-indexed sum-indices (array [[1 2 3] [4 5 6]])) (array [[0 1 2] [1 2 3]])))
      (is (equals (emap-indexed select-val (array [[1 2 3] [4 5 6]])) (array [[1 2 3] [4 5 6]])))
      (is (equals (emap-indexed i*val (array [[1 2 3] [4 5 6]])) (array [[0 2 6] [4 10 18]])))
      (is (equals (emap-indexed indices+vals (array [[1 2 3] [4 5 6]]) (array [[7 8 9] [10 11 12]]))
                  (array [[8 11 14] [15 18 21]]))))
    (testing "tensors"
      (is (equals (emap-indexed sum-indices (array [[[1 2] [3 4]] [[5 6] [7 8]]]))
                  (array [[[0 1] [1 2]] [[1 2] [2 3]]])))
      (is (equals (emap-indexed i*val (array [[[1 2] [3 4]] [[5 6] [7 8]]]))
                  (array [[[0 2] [3 8]] [[5 12] [14 24]]])))
      (is (equals (emap-indexed indices+vals
                                (array [[[1 2] [3 4]] [[5 6] [7 8]]])
                                (array [[[16 15] [14 13]] [[12 11] [10 9]]]))
                  (array [[[17 18] [18 19]] [[18 19] [19 20]]]))))))

(deftest select-range-test
  (testing "selecting from scalars"
    (is (equals (select-range (array 7)) (array 7))))
  (testing "selecting from vectors"
    (is (equals (select-range (array [1 2 3]) 1) (array 2)))
    (is (equals (select-range (array [1 2 3]) [0 1]) (array [1])))
    (is (equals (select-range (array [1 2 3]) [0 2]) (array [1 2])))
    (is (equals (select-range (array [1 2 3 4 5 6]) [0 6 2]) (array [1 3 5])))
    (is (equals (select-range (array [1 2 3]) :first) (array 1)))
    (is (equals (select-range (array [1 2 3]) :last) (array 3)))
    (is (equals (select-range (array [1 2 3]) :all) (array [1 2 3])))
    (is (equals (select-range (array [1 2 3]) :butlast) (array [1 2])))
    (is (equals (select-range (array [1 2 3]) :rest) (array [2 3]))))
  (testing "select-rangeing from matrices"
    (let [nd (array [[1 2 3] [4 5 6] [7 8 9]])]
      (is (equals (select-range nd 1 1) (array 5)))
      (is (equals (select-range nd 1 [1 3]) (array [5 6])))
      (is (equals (select-range nd [0 1] [0 2]) (array [[1 2]])))
      (is (equals (select-range nd [1 3] 1) (array [5 8])))
      (is (equals (select-range nd [1 3] [0 1]) (array [[4] [7]])))
      (is (equals (select-range nd :all :all) nd))
      (is (equals (select-range nd 2 :all) (array [7 8 9])))
      (is (equals (select-range nd :all 2) (array [3 6 9])))
      (is (equals (select-range nd :all [2 3]) (array [[3] [6] [9]])))
      (is (equals (select-range nd :all :rest) (array [[2 3] [5 6] [8 9]])))
      (is (equals (select-range nd :butlast :first) (array [1 4])))
      (is (equals (select-range nd :butlast [0 2]) (array [[1 2] [4 5]])))
      (is (equals (select-range nd :last :all) (array [7 8 9])))
      (is (equals (select-range nd :last 1) (array 8)))))
  (testing "selecting from tensors"
    (let [nd (array [[[1 2] [3 4] [5 6]]
                     [[7 8] [9 10] [11 12]]
                     [[13 14] [15 16] [17 18]]])]
      (is (equals (select-range nd 1 1 1) (array 10)))
      (is (equals (select-range nd 1 1 [1 2]) (array [10])))
      (is (equals (select-range nd 1 1 [0 2]) (array [9 10])))
      (is (equals (select-range nd [0 2] 1 [1 2]) (array [[4] [10]])))
      (is (equals (select-range nd [0 2] 1 0) (array [3 9])))
      (is (equals (select-range nd 2 [2 3] [0 2]) (array [[17 18]])))
      (is (equals (select-range nd [0 3 2] 0 1) (array [2 14])))
      (is (equals (select-range nd [0 3 2] [0 1] [1 2]) (array [[[2]] [[14]]])))
      (is (equals (select-range nd :all [0 1] [1 2]) (array [[[2]] [[8]] [[14]]])))
      (is (equals (select-range nd 2 :all :last) (array [14 16 18])))
      (is (equals (select-range nd :rest :butlast :first) (array [[7 9] [13 15]]))))))

(deftest set-range!-test
  (testing "scalars"
    (let [target (array 3)
          src (array 7)
          result (set-range! target src)]
      (is (equals target (array 7)))
      (is (equals target result))
      (is (equals src (array 7)))))
  (testing "vectors"
    (let [target (array [0 1 2 3])
          src (array 7)
          result (set-range! target 1 src)]
      (is (equals target (array [0 7 2 3])))
      (is (equals result target))
      (is (equals src (array 7))))
    (let [target (array [0 1 2 3])
          src (array [7 8])
          result (set-range! target [1 3] src)]
      (is (equals target (array [0 7 8 3])))
      (is (equals result target))
      (is (equals src (array [7 8]))))
    (is (equals (set-range! (array [1 2 3]) [1 2] (array [7])) (array [1 7 3])))
    (is (equals (set-range! (array [1 2 3 4 5 6]) [0 7 2] (array [7 8 9]))
                (array [7 2 8 4 9 6])))
    (is (equals (set-range! (array [1 2 3]) :all (array [7 8 9])) (array [7 8 9])))
    (is (equals (set-range! (array [1 2 3]) :first (array [7])) (array [7 2 3])))
    (is (equals (set-range! (array [1 2 3]) :last (array [7])) (array [1 2 7])))
    (is (equals (set-range! (array [1 2 3]) :rest (array [7 8])) (array [1 7 8])))
    (is (equals (set-range! (array [1 2 3]) :butlast (array [7 8])) (array [7 8 3]))))
  (testing "matrices"
    (let [target (array [[1 2] [3 4]])
          src (array 7)
          result (set-range! target 1 0 src)]
      (is (equals result (array [[1 2] [7 4]])))
      (is (equals result target))
      (is (equals src (array 7))))
    (let [nd (fn [] (array [[1 2 3] [4 5 6] [7 8 9]]))]
      (is (equals (set-range! (nd) 1 [1 3] (array [10 11]))
                  (array [[1 2 3] [4 10 11] [7 8 9]])))
      (is (equals (set-range! (nd) [0 1] [0 2] (array [[10 11]]))
                  (array [[10 11 3] [4 5 6] [7 8 9]])))
      (is (equals (set-range! (nd) [1 3] 1 (array [10 11]))
                  (array [[1 2 3] [4 10 6] [7 11 9]])))
      (is (equals (set-range! (nd) [1 3] [0 1] (array [[10] [11]]))
                  (array [[1 2 3] [10 5 6] [11 8 9]])))
      (is (equals (set-range! (nd) :all :all (array [[10 11 12] [13 14 15] [16 17 18]]))
                  (array [[10 11 12] [13 14 15] [16 17 18]])))
      (is (equals (set-range! (nd) 2 :all (array [10 11 12]))
                  (array [[1 2 3] [4 5 6] [10 11 12]])))
      (is (equals (set-range! (nd) :all 2 (array [10 11 12]))
                  (array [[1 2 10] [4 5 11] [7 8 12]])))
      (is (equals (set-range! (nd) :all [2 3] (array [[10] [11] [12]]))
                  (array [[1 2 10] [4 5 11] [7 8 12]])))
      (is (equals (set-range! (nd) :all :rest (array [[10 11] [12 13] [14 15]]))
                  (array [[1 10 11] [4 12 13] [7 14 15]])))
      (is (equals (set-range! (nd) :butlast :first (array [10 11]))
                  (array [[10 2 3] [11 5 6] [7 8 9]])))
      (is (equals (set-range! (nd) :butlast [0 2] (array [[10 11] [12 13]]))
                  (array [[10 11 3] [12 13 6] [7 8 9]])))
      (is (equals (set-range! (nd) :last :all (array [10 11 12]))
                  (array [[1 2 3] [4 5 6] [10 11 12]])))
      (is (equals (set-range! (nd) :last 1 (array 10))
                  (array [[1 2 3] [4 5 6] [7 10 9]])))))
  (testing "tensors"
    (let [nd (fn [] (array [[[1 2] [3 4] [5 6]]
                            [[7 8] [9 10] [11 12]]
                            [[13 14] [15 16] [17 18]]]))]
      (is (equals (set-range! (nd) 1 1 1 (array 19))
                  (array [[[1 2] [3 4] [5 6]]
                          [[7 8] [9 19] [11 12]]
                          [[13 14] [15 16] [17 18]]])))
      (is (equals (set-range! (nd) 1 1 [1 2] (array [19]))
                  (array [[[1 2] [3 4] [5 6]]
                          [[7 8] [9 19] [11 12]]
                          [[13 14] [15 16] [17 18]]])))
      (is (equals (set-range! (nd) 1 1 [0 2] (array [19 20]))
                  (array [[[1 2] [3 4] [5 6]]
                          [[7 8] [19 20] [11 12]]
                          [[13 14] [15 16] [17 18]]])))
      (is (equals (set-range! (nd) [0 2] 1 [1 2] (array [[19] [20]]))
                  (array [[[1 2] [3 19] [5 6]]
                          [[7 8] [9 20] [11 12]]
                          [[13 14] [15 16] [17 18]]])))
      (is (equals (set-range! (nd) [0 2] 1 0 (array [19 20]))
                  (array [[[1 2] [19 4] [5 6]]
                          [[7 8] [20 10] [11 12]]
                          [[13 14] [15 16] [17 18]]])))
      (is (equals (set-range! (nd) 2 [2 3] [0 2] (array [[19 20]]))
                  (array [[[1 2] [3 4] [5 6]]
                          [[7 8] [9 10] [11 12]]
                          [[13 14] [15 16] [19 20]]])))
      (is (equals (set-range! (nd) [0 3 2] 0 1 (array [19 20]))
                  (array [[[1 19] [3 4] [5 6]]
                          [[7 8] [9 10] [11 12]]
                          [[13 20] [15 16] [17 18]]])))
      (is (equals (set-range! (nd) [0 3 2] [0 1] [1 2] (array [[[19]] [[20]]]))
                  (array [[[1 19] [3 4] [5 6]]
                          [[7 8] [9 10] [11 12]]
                          [[13 20] [15 16] [17 18]]])))
      (is (equals (set-range! (nd) :all [0 1] [1 2] (array [[[19]] [[20]] [[21]]]))
                  (array [[[1 19] [3 4] [5 6]]
                          [[7 20] [9 10] [11 12]]
                          [[13 21] [15 16] [17 18]]])))
      (is (equals (set-range! (nd) 2 :all :last (array [19 20 21]))
                  (array [[[1 2] [3 4] [5 6]]
                          [[7 8] [9 10] [11 12]]
                          [[13 19] [15 20] [17 21]]])))
      (is (equals (set-range! (nd) :rest :butlast :first (array [[19 20] [21 22]]))
                  (array [[[1 2] [3 4] [5 6]]
                          [[19 8] [20 10] [11 12]]
                          [[21 14] [22 16] [17 18]]]))))))

(deftest mset!-test
  (testing "scalars"
    (let [nd (array 3)
          result (mset! nd 7)]
      (is (equals result (array 7)))
      (is (equals nd (array 7)))))
  (testing "vectors"
    (let [nd (array [1 2 3])
          result (mset! nd 1 7)]
      (is (equals result (array [1 7 3])))
      (is (equals nd (array [1 7 3])))))
  (testing "matrices"
    (let [nd (array [[1 2 3] [4 5 6]])
          result (mset! nd 0 2 7)]
      (is (equals result (array [[1 2 7] [4 5 6]])))
      (is (equals nd (array [[1 2 7] [4 5 6]])))))
  (testing "tensors"
    (let [nd (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]])
          result (mset! nd 1 0 2 13)]
      (is (equals result (array [[[1 2 3] [4 5 6]] [[7 8 13] [10 11 12]]])))
      (is (equals nd (array [[[1 2 3] [4 5 6]] [[7 8 13] [10 11 12]]]))))))

(deftest mset-test
  (testing "scalars"
    (let [nd (array 3)
          result (mset nd 7)]
      (is (equals result (array 7)))
      (is (equals nd (array 3)))))
  (testing "vectors"
    (let [nd (array [1 2 3])
          result (mset nd 1 7)]
      (is (equals result (array [1 7 3])))
      (is (equals nd (array [1 2 3])))))
  (testing "matrices"
    (let [nd (array [[1 2 3] [4 5 6]])
          result (mset nd 0 2 7)]
      (is (equals result (array [[1 2 7] [4 5 6]])))
      (is (equals nd (array [[1 2 3] [4 5 6]])))))
  (testing "tensors"
    (let [nd (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]])
          result (mset nd 1 0 2 13)]
      (is (equals result (array [[[1 2 3] [4 5 6]] [[7 8 13] [10 11 12]]])))
      (is (equals nd (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]]))))))

(deftest shift-test
  (testing "scalar"
    (let [nd (array 7)
          result (shift nd [])]
      (is (equals result (array 7)))
      (is (equals nd (array 7)))))
  (testing "vectof"
    (let [nd (array [1 2 3])
          result (shift nd 0 1)]
      (is (equals result (array [2 3 0])))
      (is (equals nd (array [1 2 3])))
      (is (equals (shift nd [1]) (array [2 3 0])))
      (is (equals (shift nd 0 -1) (array [0 1 2])))
      (is (equals (shift nd [-1]) (array [0 1 2])))
      (is (equals (shift nd [20]) (array [0 0 0])))
      (is (equals (shift nd [-3]) (array [0 0 0]))))
    (testing "matrix"
      (let [nd (array [[1 2 3] [4 5 6]])
            result (shift nd [1 -1])]
        (is (equals result (array [[0 4 5] [0 0 0]])))
        (is (equals nd (array [[1 2 3] [4 5 6]])))
        (is (equals (shift nd 0 -1) (array [[0 0 0] [1 2 3]])))
        (is (equals (shift nd 1 2) (array [[3 0 0] [6 0 0]])))
        (is (equals (shift nd 0 -10) (array [[0 0 0] [0 0 0]])))
        (is (equals (shift nd 1 3) (array [[0 0 0] [0 0 0]])))))
    (testing "tensor"
      (let [nd (array [[[1 2 3] [4 5 6]]
                       [[7 8 9] [10 11 12]]
                       [[13 14 15] [16 17 18]]])]
        (is (equals (shift nd 0 1) (array [[[7 8 9] [10 11 12]]
                                           [[13 14 15] [16 17 18]]
                                           [[0 0 0] [0 0 0]]])))
        (is (equals (shift nd [-2 0 0]) (array [[[0 0 0] [0 0 0]]
                                                [[0 0 0] [0 0 0]]
                                                [[1 2 3] [4 5 6]]])))
        (is (equals (shift nd [1 1 -2]) (array [[[0 0 10] [0 0 0]]
                                                [[0 0 16] [0 0 0]]
                                                [[0 0 0] [0 0 0]]])))))))

(deftest broadcast-test
  (is (equals (broadcast (array 7) []) (array 7)))
  (is (equals (broadcast (array 7) nil) (array 7)))
  (is (equals (broadcast (array 7) [2 3]) (array [[7 7 7] [7 7 7]])))
  (is (equals (broadcast (array 7) [2 2 3]) (array [[[7 7 7] [7 7 7]] [[7 7 7] [7 7 7]]])))
  (is (equals (broadcast (array [1 2 3]) [2 3]) (array [[1 2 3] [1 2 3]])))
  (is (equals (broadcast (array [[1 2 3]]) [2 3]) (array [[1 2 3] [1 2 3]])))
  (is (equals (broadcast (array [1 2]) [2 3]) (array [[1 1 1] [2 2 2]])))
  (is (equals (broadcast (array [[1] [2]]) [2 3]) (array [[1 1 1] [2 2 2]])))
  (is (equals (broadcast (array [1 2 3]) [2 2 3]) (array [[[1 2 3] [1 2 3]] [[1 2 3] [1 2 3]]])))
  (is (equals (broadcast (array [[1 2 3]]) [2 2 3]) (array [[[1 2 3] [1 2 3]] [[1 2 3] [1 2 3]]])))
  (is (equals (broadcast (array [[[1 2 3]]]) [2 2 3]) (array [[[1 2 3] [1 2 3]] [[1 2 3] [1 2 3]]])))
  (is (equals (broadcast (array [1 2]) [2 2 3]) (array [[[1 1 1] [2 2 2]] [[1 1 1] [2 2 2]]])))
  (is (equals (broadcast (array [[1 2]]) [2 2 3]) (array [[[1 1 1] [2 2 2]] [[1 1 1] [2 2 2]]])))
  (is (equals (broadcast (array [[[1] [2]]]) [2 2 3]) (array [[[1 1 1] [2 2 2]] [[1 1 1] [2 2 2]]])))
  (is (equals (broadcast (array [[1 2 3] [4 5 6]]) [2 2 3]) (array [[[1 2 3] [4 5 6]] [[1 2 3] [4 5 6]]])))
  (is (equals (broadcast (array [[[1 2 3] [4 5 6]]]) [2 2 3]) (array [[[1 2 3] [4 5 6]] [[1 2 3] [4 5 6]]])))
  (is (equals (broadcast (array [[[1 2 3]] [[4 5 6]]]) [2 2 3]) (array [[[1 2 3] [1 2 3]] [[4 5 6] [4 5 6]]])))
  (is (equals (broadcast (array [[1 2] [3 4]]) [2 2 3]) (array [[[1 1 1] [2 2 2]] [[3 3 3] [4 4 4]]])))
  (is (equals (broadcast (array [[[1] [2]] [[3] [4]]]) [2 2 3]) (array [[[1 1 1] [2 2 2]] [[3 3 3] [4 4 4]]]))))

(deftest rows-test
  (testing "returns views of the original data"
    (let [nd (array [[1 2 3] [4 5 6]])]
      (assign! (second (rows nd)) (array [10 11 12]))
      (is (equals nd (array [[1 2 3] [10 11 12]])))))
  (testing "returns the expected results"
    (is (thrown? Exception (rows (array 7))))
    (is (array-seqs= (rows (array [1 2 3])) [(array [1 2 3])]))
    (is (array-seqs= (rows (array [[1 2 3]])) [(array [1 2 3])]))
    (is (array-seqs= (rows (array [[1] [2] [3]]))
                     [(array [1]) (array [2]) (array [3])]))
    (is (array-seqs= (rows (array [[1 2 3] [4 5 6]]))
                     [(array [1 2 3]) (array [4 5 6])]))
    (is (array-seqs= (rows (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]]))
                     [(array [1 2 3]) (array [4 5 6]) (array [7 8 9]) (array [10 11 12])]))))

(deftest columns-test
  (testing "returns views of the original data"
    (let [nd (array [[1 2 3] [4 5 6]])]
      (assign! (second (columns nd)) (array [10 11]))
      (is (equals nd (array [[1 10 3] [4 11 6]])))))
  (testing "returns the expected results"
    (is (thrown? Exception (columns (array 7))))
    (is (array-seqs= (columns (array [1 2 3])) [(array 1) (array 2) (array 3)]))
    (is (array-seqs= (columns (array [[1 2 3]])) [(array [1]) (array [2]) (array [3])]))
    (is (array-seqs= (columns (array [[1] [2] [3]]))
                     [(array [1 2 3])]))
    (is (array-seqs= (columns (array [[1 2 3] [4 5 6]]))
                     [(array [1 4]) (array [2 5]) (array [3 6])]))
    (is (array-seqs= (columns (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]]))
                     [(array [1 4]) (array [2 5]) (array [3 6]) (array [7 10]) (array [8 11]) (array [9 12])]))))

(deftest slices-test
  (testing "returns views of the original data"
    (let [nd (array [[1 2 3] [4 5 6]])]
      (assign! (first (slices nd 0)) (array [[10 11 12]]))
      (is (equals nd (array [[10 11 12] [4 5 6]])))))
  (testing "returns the expected results"
    (is (thrown? Exception (slices (array 7))))
    (is (array-seqs= (slices (array [1 2 3])) [(array 1) (array 2) (array 3)]))
    (is (array-seqs= (slices (array [1 2 3]) 0) [(array 1) (array 2) (array 3)]))
    (is (array-seqs= (slices (array [[1 2 3]]) 0) [(array [1 2 3])]))
    (is (array-seqs= (slices (array [[1 2 3]]) 1) [(array [1]) (array [2]) (array [3])]))
    (is (array-seqs= (slices (array [[1] [2] [3]]) 0) [(array [1]) (array [2]) (array [3])]))
    (is (array-seqs= (slices (array [[1] [2] [3]]) 1) [(array [1 2 3])]))
    (is (array-seqs= (slices (array [[1 2 3] [4 5 6]])) [(array [1 2 3]) (array [4 5 6])]))
    (is (array-seqs= (slices (array [[1 2 3] [4 5 6]]) 0) [(array [1 2 3]) (array [4 5 6])]))
    (is (array-seqs= (slices (array [[1 2 3] [4 5 6]]) 1) [(array [1 4]) (array [2 5]) (array [3 6])]))
    (is (array-seqs= (slices (array [[[1 2 3] [4 5 6]]
                                     [[7 8 9] [10 11 12]]]) 0)
                     [(array [[1 2 3] [4 5 6]]) (array [[7 8 9] [10 11 12]])]))
    (is (array-seqs= (slices (array [[[1 2 3] [4 5 6]]
                                     [[7 8 9] [10 11 12]]]) 1)
                     [(array [[1 2 3] [7 8 9]]) (array [[4 5 6] [10 11 12]])]))
    (is (array-seqs= (slices (array [[[1 2 3] [4 5 6]]
                                     [[7 8 9] [10 11 12]]]) 2)
                     [(array [[1 4] [7 10]]) (array [[2 5] [8 11]]) (array [[3 6] [9 12]])]))))

(deftest submatrix-test
  (testing "returns views of the original data"
    (let [nd (array [[1 2 3] [4 5 6]])]
      (assign! (submatrix nd [[0 1] [1 2]]) (array [[10 11]]))
      (is (equals nd (array [[1 10 11] [4 5 6]])))))
  (testing "produces expected subtensors"
    (is (equals (submatrix (array 7) []) (array 7)))
    (is (thrown? Exception (submatrix (array 7) [[0 1]]) (array 7)))
    (is (equals (submatrix (array [1 2 3]) [[0 1]]) (array [1])))
    (is (equals (submatrix (array [1 2 3]) [[0 2]]) (array [1 2])))
    (is (equals (submatrix (array [1 2 3]) 0 [1 2]) (array [2 3])))
    (is (equals (submatrix (array [1 2 3]) [[0 33]]) (array [1 2 3])))
    (is (equals (submatrix (array [[1 2 3] [4 5 6]]) [[0 1]]) (array [[1 2 3]])))
    (is (equals (submatrix (array [[1 2 3] [4 5 6]]) 0 [0 1]) (array [[1 2 3]])))
    (is (equals (submatrix (array [[1 2 3] [4 5 6]]) [nil [0 2]]) (array [[1 2] [4 5]])))
    (is (equals (submatrix (array [[1 2 3] [4 5 6]]) 1 [0 2]) (array [[1 2] [4 5]])))
    (is (equals (submatrix (array [[1 2 3] [4 5 6]]) [[0 1] [1 2]]) (array [[2 3]])))
    (is (equals (submatrix (array [[1 2 3] [4 5 6]]) 0 1 1 2) (array [[2 3]])))
    (is (equals (submatrix (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]]) [[1 1]])
                (array [[[7 8 9] [10 11 12]]])))
    (is (equals (submatrix (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]]) 0 [1 1])
                (array [[[7 8 9] [10 11 12]]])))
    (is (equals (submatrix (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]]) [nil [1 1]])
                (array [[[4 5 6]] [[10 11 12]]])))
    (is (equals (submatrix (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]]) [nil [1 1] nil])
                (array [[[4 5 6]] [[10 11 12]]])))
    (is (equals (submatrix (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]]) 1 [1 1])
                (array [[[4 5 6]] [[10 11 12]]])))
    (is (equals (submatrix (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]]) [nil nil [1 2]])
                (array [[[2 3] [5 6]] [[8 9] [11 12]]])))
    (is (equals (submatrix (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]]) 2 [1 2])
                (array [[[2 3] [5 6]] [[8 9] [11 12]]])))
    (is (equals (submatrix (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]]) [[1 1] [1 1]])
                (array [[[10 11 12]]])))
    (is (equals (submatrix (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]]) [[1 1] [1 1] nil])
                (array [[[10 11 12]]])))
    (is (equals (submatrix (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]]) [nil [1 1] [1 2]])
                (array [[[5 6]] [[11 12]]])))
    (is (equals (submatrix (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]]) [[1 1] [1 1] [0 2]])
                (array [[[10 11]]])))))

(deftest join-along-test
  (testing "scalars"
    (is (thrown? Exception (join-along 0 (array 3) (array 7))))
    (is (thrown? Exception (join-along 0 (array [1 2]) (array 7)))))
  (testing "vectors"
    (is (equals (join-along 0 (array [1 2]) (array [3 4])) (array [1 2 3 4])))
    (is (thrown? Exception (join-along 1 (array [1 2]) (array [3 4])))))
  (testing "matrices"
    (is (equals (join-along 0 (array [[1 2] [3 4] [5 6]]) (array [[7 8] [9 10]]))
                (array [[1 2] [3 4] [5 6] [7 8] [9 10]])))
    (is (thrown? Exception (join-along 1 (array [[1 2] [3 4] [5 6]]) (array [[5 6] [7 8]]))))
    (is (equals (join-along 1 (array [[1 2] [3 4]]) (array [[5 6] [7 8]]))
                (array [[1 2 5 6] [3 4 7 8]]))))
  (testing "tensors"
    (is (equals (join-along 0 (array [[[1 2] [3 4]] [[5 6] [7 8]]])
                            (array [[[9 10] [11 12]] [[13 14] [15 16]]]))
                (array [[[1 2] [3 4]] [[5 6] [7 8]] [[9 10] [11 12]] [[13 14] [15 16]]])))
    (is (equals (join-along 1 (array [[[1 2] [3 4]] [[5 6] [7 8]]])
                            (array [[[9 10] [11 12]] [[13 14] [15 16]]]))
                (array [[[1 2] [3 4] [9 10] [11 12]] [[5 6] [7 8] [13 14] [15 16]]])))
    (is (equals (join-along 2 (array [[[1 2] [3 4]] [[5 6] [7 8]]])
                            (array [[[9 10] [11 12]] [[13 14] [15 16]]]))
                (array [[[1 2 9 10] [3 4 11 12]] [[5 6 13 14] [7 8 15 16]]]))))
  (testing "copies data to a new tensor"
    (let [a (array [[1 2] [3 4] [5 6]])
          result (join-along 0 a (array [[7 8] [9 10]]))]
      (assign! a (array [[10 11] [12 13] [14 15]]))
      ; Checks the assignment to a worked.
      (is (equals a (array [[10 11] [12 13] [14 15]])))
      ; Checks the assignment did not affect `result`.
      (is (equals result (array [[1 2] [3 4] [5 6] [7 8] [9 10]])))
      (assign! result (array [[16 17] [18 19] [20 21] [22 23] [24 25]]))
      ; Checks the assignment to `result` worked.
      (is (equals result (array [[16 17] [18 19] [20 21] [22 23] [24 25]])))
      ; Checks the assignment to `result` did not affect `a`.
      (is (equals a (array [[10 11] [12 13] [14 15]]))))))

(deftest equals-test
  (is (true? (equals (array [1 2 3]) (array [1 2 3]))))
  (is (false? (equals (array [1 2 3]) (array [1 2 4]))))
  (is (false? (equals (array [1 2 4]) (array [1 2 3]))))
  (is (true? (equals (array [1 2 3]) (array [1 2 (+ 3 1e-3)]) 1e-2)))
  (is (false? (equals (array [1 2 3]) (array [1 2 (+ 3 1e-3)]) 1e-4)))
  (is (false? (equals (array [1 2 3]) (array [1 2]))))
  (is (false? (equals (array 1) (array [1 2]))))
  (is (false? (equals (array [1 2]) (array [[1 2]]))))
  (is (false? (equals (array [1 2]) (array [[1] [2]]))))
  (is (true? (equals (array [[1 2]]) (array [[1 2]]))))
  (is (false? (equals (array [[1 2]]) (array [[1] [2]]))))
  (is (true? (equals (array [[1] [2]]) (array [[1] [2]]))))
  (is (false? (equals (array [[1] [2]]) (array [[1] [2] [3]]))))
  (is (true? (equals (array [[1 2] [3 4]]) (array [[1 2] [3 4]]))))
  (is (false? (equals (array [[1 10] [3 4]]) (array [[1 2] [3 4]]))))
  (is (true? (equals (array [[1 2.001] [3 4]]) (array [[1 2] [3 4]]) 1e-2)))
  (is (false? (equals (array [[1 2.001] [3 4]]) (array [[1 2] [3 4]]) 1e-4)))
  (is (false? (equals (array [[1 2] [3 4]]) (array [[1 3] [2 4]]))))
  (is (false? (equals (array [[1 2] [3 4]]) (array [1 2 3 4]))))
  (is (true? (equals (array [[[1 2] [3 4]] [[5 6] [7 8]]])
                     (array [[[1 2] [3 4]] [[5 6] [7 8]]]))))
  (is (false? (equals (array [[[1 2] [3 4]] [[5 6] [7 8]]])
                      (array [[[1 2] [3 4]]]))))
  (is (false? (equals (array [[[1 2] [3 4]] [[5 7] [7 8]]])
                      (array [[[1 2] [3 4]] [[5 6] [7 8]]])))))

(deftest mul!-test
  (is (thrown? Exception (mul! (array 2) (array 3))))
  (is (thrown? Exception (mul! (array 2))))
  (let [a (array [1 2 3])]
    (add! a)
    (is (equals a (array [1 2 3]))))
  (let [a (array [1 2 3])
        b (array [4 5 6])
        result (mul! a b)]
    (is (equals result (array [4 10 18])))
    (is (equals a result))
    (is (equals b (array [4 5 6])))
    (is (= (rank result) 1)))
  (let [a (array [1 2 3])
        b (array [4 5 6])]
    (mul! a (array 2) b)
    (is (equals a (array [8 20 36])))
    (is (equals b (array [4 5 6]))))
  (let [a (array [1 2 3])
        b (array [4 5 6])]
    (mul! a b (array 2))
    (is (equals a (array [8 20 36])))
    (is (equals b (array [4 5 6]))))
  (let [a (array [[1 2 3] [4 5 6]])
        b (array [3 2 1])]
    (mul! a b)
    (is (equals a (array [[3 4 3] [12 10 6]])))
    (is (= b (array [3 2 1])))
    (is (= (rank a) 2))
    (is (thrown? Exception (mul! b a))))
  (let [a (array [[1 2 3] [4 5 6]])
        b (array [[3 2 1]])]
    (mul! a b)
    (is (equals a (array [[3 4 3] [12 10 6]])))
    (is (equals b (array [[3 2 1]])))
    (is (thrown? Exception (mul! b a))))
  (let [a (array [[1 2 3] [4 5 6]])
        b (array [2 3])]
    (mul! a b)
    (is (equals a (array [[2 4 6] [12 15 18]])))
    (is (equals b (array [2 3])))
    (is (thrown? Exception (mul! b a))))
  (let [a (array [[1 2 3] [4 5 6]])
        b (array [[2] [3]])]
    (mul! a b)
    (is (equals a (array [[2 4 6] [12 15 18]]))))
  (let [a (array [[1 2 3] [4 5 6]])
        b (array [[3 2 1]])
        c (array [2 3])]
    (mul! a b c)
    (is (equals a (array [[6 8 6] [36 30 18]])))
    (is (equals b (array [[3 2 1]])))
    (is (equals c (array [2 3])))
    (is (thrown? Exception (mul! c b a))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (mul! a (array [2 3]) (array [[3 2 1]]))
    (is (equals a (array [[6 8 6] [36 30 18]]))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (mul! a (array [[3 2 1]]) (array [[2] [3]]))
    (is (equals a (array [[6 8 6] [36 30 18]]))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (mul! a (array [[2] [3]]) (array [[3 2 1]]))
    (is (equals a (array [[6 8 6] [36 30 18]]))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (mul! a (array [3 2 1]) (array [[2] [3]]))
    (is (equals a (array [[6 8 6] [36 30 18]]))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (mul! a (array [[2] [3]]) (array [3 2 1]))
    (is (equals a (array [[6 8 6] [36 30 18]]))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (mul! a (array [2 3]) (array [3 3]))
    (is (equals a (array [[6 12 18] [36 45 54]]))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (mul! a (array [1 2 3]) (array 2))
    (is (equals a (array [[2 8 18] [8 20 36]]))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (mul! a (array 2) (array [1 2 3]))
    (is (equals a (array [[2 8 18] [8 20 36]]))))
  (let [a (array [[1 2 3] [4 5 6]])
        b (array [[0 2 1] [4 5 3]])]
    (mul! a b)
    (is (equals a (array [[0 4 3] [16 25 18]])))
    (is (equals b (array [[0 2 1] [4 5 3]]))))
  (let [a (array [[1 2 3] [4 5 6]])
        result (mul! a (array [[0 2 1] [4 5 3]]) (array [1 2 3]))]
    (is (equals a (array [[0 8 9] [16 50 54]])))
    (is (equals result (array [[0 8 9] [16 50 54]]))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (mul! a (array [1 2 3]) (array [[0 2 1] [4 5 3]]))
    (is (equals a (array [[0 8 9] [16 50 54]]))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (mul! a (array 2) (array [1 2 3]) (array [[0 2 1] [4 5 3]]))
    (is (equals a (array [[0 16 18] [32 100 108]]))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (mul! a (array [1 2 3]) (array 2) (array [[0 2 1] [4 5 3]]))
    (is (equals a (array [[0 16 18] [32 100 108]]))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (mul! a (array [1 2 3]) (array [[0 2 1] [4 5 3]]) (array 2))
    (is (equals a (array [[0 16 18] [32 100 108]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (mul! a (array [1 2 3]))
    (is (equals a (array [[[1 4 9] [4 10 18]]
                          [[7 16 27] [10 22 36]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (mul! a (array [[1 2 3]]))
    (is (equals a (array [[[1 4 9] [4 10 18]]
                          [[7 16 27] [10 22 36]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (mul! a (array [[[1 2 3]]]))
    (is (equals a (array [[[1 4 9] [4 10 18]]
                          [[7 16 27] [10 22 36]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (mul! a (array [[1 2] [3 4]]))
    (is (equals a (array [[[1 2 3] [8 10 12]]
                          [[21 24 27] [40 44 48]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (mul! a (array [[[1] [2]] [[3] [4]]]))
    (is (equals a (array [[[1 2 3] [8 10 12]]
                          [[21 24 27] [40 44 48]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (mul! a (array [[3 2 1] [6 5 4]]))
    (is (equals a (array [[[3 4 3] [24 25 24]]
                          [[21 16 9] [60 55 48]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (mul! a (array [[[3 2 1] [6 5 4]]]))
    (is (equals a (array [[[3 4 3] [24 25 24]]
                          [[21 16 9] [60 55 48]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (mul! a (array [[[3 2 1]] [[6 5 4]]]))
    (is (equals a (array [[[3 4 3] [12 10 6]]
                          [[42 40 36] [60 55 48]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (mul! a (array 2) (array [[[3 2 1]] [[6 5 4]]]))
    (is (equals a (array [[[6 8 6] [24 20 12]]
                          [[84 80 72] [120 110 96]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (mul! a (array [[[3 2 1]] [[6 5 4]]]) (array 2))
    (is (equals a (array [[[6 8 6] [24 20 12]]
                          [[84 80 72] [120 110 96]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (mul! a (array [[[3 2 1]] [[6 5 4]]]) (array [0 1 0]) (array 2))
    (is (equals a (array [[[0 8 0] [0 20 0]]
                          [[0 80 0] [0 110 0]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (mul! a (array [0 1 0]) (array [[[3 2 1]] [[6 5 4]]]) (array 2))
    (is (equals a (array [[[0 8 0] [0 20 0]]
                          [[0 80 0] [0 110 0]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (mul! a (array 2) (array [0 1 0]) (array [[[3 2 1]] [[6 5 4]]]))
    (is (equals a (array [[[0 8 0] [0 20 0]]
                          [[0 80 0] [0 110 0]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (mul! a (array [[[3 2 1]] [[6 5 4]]]) (array [[1 2] [3 4]]))
    (is (equals a (array [[[3 4 3] [24 20 12]]
                          [[126 120 108] [240 220 192]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])
        b (array [[1 2] [3 4]])
        c (array [[[3 2 1]] [[6 5 4]]])
        result (mul! a b c)]
    (is (equals result (array [[[3 4 3] [24 20 12]]
                               [[126 120 108] [240 220 192]]])))
    (is (equals a result))
    (is (equals b (array [[1 2] [3 4]])))
    (is (equals c (array [[[3 2 1]] [[6 5 4]]])))))

(deftest mul-test
  (let [a (array 2)
        b (array 3)
        result (mul a b)]
    (is (equals result (array 6)))
    (is (equals a (array 2)))
    (is (equals b (array 3)))
    (is (= (rank result) 0)))
  (is (equals (mul (array 2) (array 3) (array 4)) (array 24)))
  (is (equals (mul) (array 1)))
  (is (equals (mul (array 7)) (array 7)))
  (let [a (array [1 2 3])
        b (array [4 5 6])
        result (mul a b)]
    (is (equals result (array [4 10 18])))
    (is (equals a (array [1 2 3])))
    (is (equals b (array [4 5 6])))
    (is (= (rank result) 1)))
  (let [result (mul (array [1 2 3]) (array 2) (array [4 5 6]))]
    (is (equals result (array [8 20 36])))
    (is (= (rank result) 1)))
  (let [result (mul (array 2) (array [1 2 3]) (array [4 5 6]))]
    (is (equals result (array [8 20 36])))
    (is (= (rank result) 1)))
  (let [result (mul (array [1 2 3]) (array [4 5 6]) (array 2))]
    (is (equals result (array [8 20 36])))
    (is (= (rank result) 1)))
  (let [result (mul (array [[1 2 3] [4 5 6]]) (array [3 2 1]))]
    (is (equals result (array [[3 4 3] [12 10 6]])))
    (is (= (rank result) 2)))
  (let [result (mul (array [3 2 1]) (array [[1 2 3] [4 5 6]]))]
    (is (equals result (array [[3 4 3] [12 10 6]])))
    (is (= (rank result) 2)))
  (is (equals (mul (array [[1 2 3] [4 5 6]]) (array [[3 2 1]]))
              (array [[3 4 3] [12 10 6]])))
  (is (equals (mul (array [[3 2 1]]) (array [[1 2 3] [4 5 6]]))
              (array [[3 4 3] [12 10 6]])))
  (is (equals (mul (array [[1 2 3] [4 5 6]]) (array [2 3]))
              (array [[2 4 6] [12 15 18]])))
  (is (equals (mul (array [2 3]) (array [[1 2 3] [4 5 6]]))
              (array [[2 4 6] [12 15 18]])))
  (is (equals (mul (array [[1 2 3] [4 5 6]]) (array [[2] [3]]))
              (array [[2 4 6] [12 15 18]])))
  (is (equals (mul (array [[2] [3]]) (array [[1 2 3] [4 5 6]]))
              (array [[2 4 6] [12 15 18]])))
  (is (equals (mul (array [[1 2 3] [4 5 6]]) (array [[3 2 1]]) (array [2 3]))
              (array [[6 8 6] [36 30 18]])))
  (is (equals (mul (array [2 3]) (array [[1 2 3] [4 5 6]]) (array [[3 2 1]]))
              (array [[6 8 6] [36 30 18]])))
  (is (equals (mul (array [[1 2 3] [4 5 6]]) (array [2 3]) (array [[3 2 1]]))
              (array [[6 8 6] [36 30 18]])))
  (is (thrown? Exception (mul (array [2 3]) (array [[3 2 1]]) (array [[1 2 3] [4 5 6]]))))
  (is (equals (mul (array [[1 2 3] [4 5 6]]) (array [[3 2 1]]) (array [[2] [3]]))
              (array [[6 8 6] [36 30 18]])))
  (is (equals (mul (array [[2] [3]]) (array [[1 2 3] [4 5 6]]) (array [[3 2 1]]))
              (array [[6 8 6] [36 30 18]])))
  (is (equals (mul (array [[1 2 3] [4 5 6]]) (array [[2] [3]]) (array [[3 2 1]]))
              (array [[6 8 6] [36 30 18]])))
  (is (equals (mul (array [[1 2 3] [4 5 6]]) (array [3 2 1]) (array [[2] [3]]))
              (array [[6 8 6] [36 30 18]])))
  (is (equals (mul (array [[2] [3]]) (array [[1 2 3] [4 5 6]]) (array [3 2 1]))
              (array [[6 8 6] [36 30 18]])))
  (is (equals (mul (array [[1 2 3] [4 5 6]]) (array [[2] [3]]) (array [3 2 1]))
              (array [[6 8 6] [36 30 18]])))
  (is (equals (mul (array [[1 2 3] [4 5 6]]) (array [2 3]) (array [3 3]))
              (array [[6 12 18] [36 45 54]])))
  (is (equals (mul (array [3 3]) (array [[1 2 3] [4 5 6]]) (array [2 3]))
              (array [[6 12 18] [36 45 54]])))
  (is (equals (mul (array [3 3]) (array [2 3]) (array [[1 2 3] [4 5 6]]))
              (array [[6 12 18] [36 45 54]])))
  (is (equals (mul (array [[1 2 3] [4 5 6]]) (array [1 2 3]) (array 2))
              (array [[2 8 18] [8 20 36]])))
  (is (equals (mul (array [[1 2 3] [4 5 6]]) (array 2) (array [1 2 3]))
              (array [[2 8 18] [8 20 36]])))
  (is (equals (mul (array 2) (array [[1 2 3] [4 5 6]]) (array [1 2 3]))
              (array [[2 8 18] [8 20 36]])))
  (is (equals (mul (array 2) (array [1 2 3]) (array [[1 2 3] [4 5 6]]))
              (array [[2 8 18] [8 20 36]])))
  (is (equals (mul (array [[1 2 3] [4 5 6]]) (array [[0 2 1] [4 5 3]]))
              (array [[0 4 3] [16 25 18]])))
  (is (equals (mul (array [[1 2 3] [4 5 6]]) (array [[0 2 1] [4 5 3]]) (array [1 2 3]))
              (array [[0 8 9] [16 50 54]])))
  (is (equals (mul (array [[1 2 3] [4 5 6]]) (array [1 2 3]) (array [[0 2 1] [4 5 3]]))
              (array [[0 8 9] [16 50 54]])))
  (is (equals (mul (array [1 2 3]) (array [[1 2 3] [4 5 6]]) (array [[0 2 1] [4 5 3]]))
              (array [[0 8 9] [16 50 54]])))
  (is (equals (mul (array 2) (array [1 2 3]) (array [[1 2 3] [4 5 6]]) (array [[0 2 1] [4 5 3]]))
              (array [[0 16 18] [32 100 108]])))
  (is (equals (mul (array [1 2 3]) (array 2) (array [[1 2 3] [4 5 6]]) (array [[0 2 1] [4 5 3]]))
              (array [[0 16 18] [32 100 108]])))
  (is (equals (mul (array [1 2 3]) (array [[1 2 3] [4 5 6]]) (array 2) (array [[0 2 1] [4 5 3]]))
              (array [[0 16 18] [32 100 108]])))
  (is (equals (mul (array [1 2 3]) (array [[1 2 3] [4 5 6]]) (array [[0 2 1] [4 5 3]]) (array 2))
              (array [[0 16 18] [32 100 108]])))
  (is (equals (mul (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [1 2 3]))
              (array [[[1 4 9] [4 10 18]]
                      [[7 16 27] [10 22 36]]])))
  (is (equals (mul (array [1 2 3])
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]]))
              (array [[[1 4 9] [4 10 18]]
                      [[7 16 27] [10 22 36]]])))
  (is (equals (mul (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[1 2 3]]))
              (array [[[1 4 9] [4 10 18]]
                      [[7 16 27] [10 22 36]]])))
  (is (equals (mul (array [[1 2 3]])
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]]))
              (array [[[1 4 9] [4 10 18]]
                      [[7 16 27] [10 22 36]]])))
  (is (equals (mul (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[[1 2 3]]]))
              (array [[[1 4 9] [4 10 18]]
                      [[7 16 27] [10 22 36]]])))
  (is (equals (mul (array [[[1 2 3]]])
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]]))
              (array [[[1 4 9] [4 10 18]]
                      [[7 16 27] [10 22 36]]])))
  (is (equals (mul (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[1 2] [3 4]]))
              (array [[[1 2 3] [8 10 12]]
                      [[21 24 27] [40 44 48]]])))
  (is (equals (mul (array [[1 2] [3 4]])
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]]))
              (array [[[1 2 3] [8 10 12]]
                      [[21 24 27] [40 44 48]]])))
  (is (equals (mul (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[[1] [2]] [[3] [4]]]))
              (array [[[1 2 3] [8 10 12]]
                      [[21 24 27] [40 44 48]]])))
  (is (equals (mul (array [[[1] [2]] [[3] [4]]])
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]]))
              (array [[[1 2 3] [8 10 12]]
                      [[21 24 27] [40 44 48]]])))
  (is (equals (mul (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[3 2 1] [6 5 4]]))
              (array [[[3 4 3] [24 25 24]]
                      [[21 16 9] [60 55 48]]])))
  (is (equals (mul (array [[3 2 1] [6 5 4]])
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]]))
              (array [[[3 4 3] [24 25 24]]
                      [[21 16 9] [60 55 48]]])))
  (is (equals (mul (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[[3 2 1] [6 5 4]]]))
              (array [[[3 4 3] [24 25 24]]
                      [[21 16 9] [60 55 48]]])))
  (is (equals (array [[[3 4 3] [24 25 24]]
                      [[21 16 9] [60 55 48]]])
              (mul (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[[3 2 1] [6 5 4]]]))))
  (is (equals (mul (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[[3 2 1]] [[6 5 4]]]))
              (array [[[3 4 3] [12 10 6]]
                      [[42 40 36] [60 55 48]]])))
  (is (equals (mul (array [[[3 2 1]] [[6 5 4]]])
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]]))
              (array [[[3 4 3] [12 10 6]]
                      [[42 40 36] [60 55 48]]])))
  (is (equals (mul (array 2)
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[[3 2 1]] [[6 5 4]]]))
              (array [[[6 8 6] [24 20 12]]
                      [[84 80 72] [120 110 96]]])))
  (is (equals (mul (array 2)
                   (array [[[3 2 1]] [[6 5 4]]])
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]]))
              (array [[[6 8 6] [24 20 12]]
                      [[84 80 72] [120 110 96]]])))
  (is (equals (mul (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[[3 2 1]] [[6 5 4]]])
                   (array [0 1 0])
                   (array 2))
              (array [[[0 8 0] [0 20 0]]
                      [[0 80 0] [0 110 0]]])))
  (is (equals (mul (array [0 1 0])
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[[3 2 1]] [[6 5 4]]])
                   (array 2))
              (array [[[0 8 0] [0 20 0]]
                      [[0 80 0] [0 110 0]]])))
  (is (equals (mul (array [0 1 0])
                   (array [[[3 2 1]] [[6 5 4]]])
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array 2))
              (array [[[0 8 0] [0 20 0]]
                      [[0 80 0] [0 110 0]]])))
  (is (equals (mul (array [0 1 0])
                   (array [[[3 2 1]] [[6 5 4]]])
                   (array 2)
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]]))
              (array [[[0 8 0] [0 20 0]]
                      [[0 80 0] [0 110 0]]])))
  (is (equals (mul (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[[3 2 1]] [[6 5 4]]])
                   (array [[1 2] [3 4]]))
              (array [[[3 4 3] [24 20 12]]
                      [[126 120 108] [240 220 192]]])))
  (is (equals (mul (array [[[3 2 1]] [[6 5 4]]])
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[1 2] [3 4]]))
              (array [[[3 4 3] [24 20 12]]
                      [[126 120 108] [240 220 192]]])))
  (let [a (array [[1 2] [3 4]])
        b (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])
        c (array [[[3 2 1]] [[6 5 4]]])]
    (is (equals (mul a b c) (array [[[3 4 3] [24 20 12]]
                                    [[126 120 108] [240 220 192]]])))
    (is (equals a (array [[1 2] [3 4]])))
    (is (equals b (array [[[1 2 3] [4 5 6]]
                          [[7 8 9] [10 11 12]]])))
    (is (equals c (array [[[3 2 1]] [[6 5 4]]])))))

(deftest add!-test
  (is (thrown? Exception (add! (array 2) (array 3))))
  (is (thrown? Exception (add! (array 2))))
  (let [a (array [1 2 3])]
    (add! a)
    (is (equals a (array [1 2 3]))))
  (let [a (array [1 2 3])
        b (array [4 5 6])
        result (add! a b)]
    (is (equals result (array [5 7 9])))
    (is (equals a result))
    (is (equals b (array [4 5 6])))
    (is (= (rank result) 1)))
  (let [a (array [1 2 3])
        b (array [4 5 6])]
    (add! a (array 2) b)
    (is (equals a (array [7 9 11])))
    (is (equals b (array [4 5 6]))))
  (let [a (array [1 2 3])
        b (array [4 5 6])]
    (add! a b (array 2))
    (is (equals a (array [7 9 11])))
    (is (equals b (array [4 5 6]))))
  (let [a (array [[1 2 3] [4 5 6]])
        b (array [3 2 1])]
    (add! a b)
    (is (equals a (array [[4 4 4] [7 7 7]])))
    (is (= b (array [3 2 1])))
    (is (= (rank a) 2))
    (is (thrown? Exception (add! b a))))
  (let [a (array [[1 2 3] [4 5 6]])
        b (array [[3 2 1]])]
    (add! a b)
    (is (equals a (array [[4 4 4] [7 7 7]])))
    (is (equals b (array [[3 2 1]])))
    (is (thrown? Exception (add! b a))))
  (let [a (array [[1 2 3] [4 5 6]])
        b (array [2 3])]
    (add! a b)
    (is (equals a (array [[3 4 5] [7 8 9]])))
    (is (equals b (array [2 3])))
    (is (thrown? Exception (add! b a))))
  (let [a (array [[1 2 3] [4 5 6]])
        b (array [[2] [3]])]
    (add! a b)
    (is (equals a (array [[3 4 5] [7 8 9]]))))
  (let [a (array [[1 2 3] [4 5 6]])
        b (array [[3 2 1]])
        c (array [2 3])]
    (add! a b c)
    (is (equals a (array [[6 6 6] [10 10 10]])))
    (is (equals b (array [[3 2 1]])))
    (is (equals c (array [2 3])))
    (is (thrown? Exception (add! c b a))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (add! a (array [2 3]) (array [[3 2 1]]))
    (is (equals a (array [[6 6 6] [10 10 10]]))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (add! a (array [[3 2 1]]) (array [[2] [3]]))
    (is (equals a (array [[6 6 6] [10 10 10]]))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (add! a (array [[2] [3]]) (array [[3 2 1]]))
    (is (equals a (array [[6 6 6] [10 10 10]]))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (add! a (array [3 2 1]) (array [[2] [3]]))
    (is (equals a (array [[6 6 6] [10 10 10]]))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (add! a (array [[2] [3]]) (array [3 2 1]))
    (is (equals a (array [[6 6 6] [10 10 10]]))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (add! a (array [2 3]) (array [3 3]))
    (is (equals a (array [[6 7 8] [10 11 12]]))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (add! a (array [1 2 3]) (array 2))
    (is (equals a (array [[4 6 8] [7 9 11]]))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (add! a (array 2) (array [1 2 3]))
    (is (equals a (array [[4 6 8] [7 9 11]]))))
  (let [a (array [[1 2 3] [4 5 6]])
        b (array [[0 2 1] [4 5 3]])]
    (add! a b)
    (is (equals a (array [[1 4 4] [8 10 9]])))
    (is (equals b (array [[0 2 1] [4 5 3]]))))
  (let [a (array [[1 2 3] [4 5 6]])
        result (add! a (array [[0 2 1] [4 5 3]]) (array [1 2 3]))]
    (is (equals a (array [[2 6 7] [9 12 12]])))
    (is (equals result (array [[2 6 7] [9 12 12]]))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (add! a (array [1 2 3]) (array [[0 2 1] [4 5 3]]))
    (is (equals a (array [[2 6 7] [9 12 12]]))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (add! a (array 2) (array [1 2 3]) (array [[0 2 1] [4 5 3]]))
    (is (equals a (array [[4 8 9] [11 14 14]]))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (add! a (array [1 2 3]) (array 2) (array [[0 2 1] [4 5 3]]))
    (is (equals a (array [[4 8 9] [11 14 14]]))))
  (let [a (array [[1 2 3] [4 5 6]])]
    (add! a (array [1 2 3]) (array [[0 2 1] [4 5 3]]) (array 2))
    (is (equals a (array [[4 8 9] [11 14 14]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (add! a (array [1 2 3]))
    (is (equals a (array [[[2 4 6] [5 7 9]]
                          [[8 10 12] [11 13 15]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (add! a (array [[1 2 3]]))
    (is (equals a (array [[[2 4 6] [5 7 9]]
                          [[8 10 12] [11 13 15]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (add! a (array [[[1 2 3]]]))
    (is (equals a (array [[[2 4 6] [5 7 9]]
                          [[8 10 12] [11 13 15]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (add! a (array [[1 2] [3 4]]))
    (is (equals a (array [[[2 3 4] [6 7 8]]
                          [[10 11 12] [14 15 16]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (add! a (array [[[1] [2]] [[3] [4]]]))
    (is (equals a (array (array [[[2 3 4] [6 7 8]]
                                 [[10 11 12] [14 15 16]]])))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (add! a (array [[3 2 1] [6 5 4]]))
    (is (equals a (array [[[4 4 4] [10 10 10]] [[10 10 10] [16 16 16]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (add! a (array [[[3 2 1] [6 5 4]]]))
    (is (equals a (array [[[4 4 4] [10 10 10]] [[10 10 10] [16 16 16]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (add! a (array [[[3 2 1]] [[6 5 4]]]))
    (is (equals a (array [[[4 4 4] [7 7 7]] [[13 13 13] [16 16 16]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (add! a (array 2) (array [[[3 2 1]] [[6 5 4]]]))
    (is (equals a (array [[[6 6 6] [9 9 9]] [[15 15 15] [18 18 18]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (add! a (array [[[3 2 1]] [[6 5 4]]]) (array 2))
    (is (equals a (array [[[6 6 6] [9 9 9]] [[15 15 15] [18 18 18]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (add! a (array [[[3 2 1]] [[6 5 4]]]) (array [0 1 0]) (array 2))
    (is (equals a (array [[[6 7 6] [9 10 9]] [[15 16 15] [18 19 18]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (add! a (array [0 1 0]) (array [[[3 2 1]] [[6 5 4]]]) (array 2))
    (is (equals a (array [[[6 7 6] [9 10 9]] [[15 16 15] [18 19 18]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (add! a (array 2) (array [0 1 0]) (array [[[3 2 1]] [[6 5 4]]]))
    (is (equals a (array [[[6 7 6] [9 10 9]] [[15 16 15] [18 19 18]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (add! a (array [[[3 2 1]] [[6 5 4]]]) (array [[1 2] [3 4]]))
    (is (equals a (array [[[5 5 5] [9 9 9]] [[16 16 16] [20 20 20]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])
        b (array [[1 2] [3 4]])
        c (array [[[3 2 1]] [[6 5 4]]])
        result (add! a b c)]
    (is (equals result (array [[[5 5 5] [9 9 9]] [[16 16 16] [20 20 20]]])))
    (is (equals a result))
    (is (equals b (array [[1 2] [3 4]])))
    (is (equals c (array [[[3 2 1]] [[6 5 4]]])))))

(deftest add-test
  (let [a (array 2)
        b (array 3)
        result (add a b)]
    (is (equals result (array 5)))
    (is (equals a (array 2)))
    (is (equals b (array 3)))
    (is (= (rank result) 0)))
  (is (equals (add (array 2) (array 3) (array 4)) (array 9)))
  (is (equals (add) (array 0)))
  (is (equals (add (array 7)) (array 7)))
  (let [a (array [1 2 3])
        b (array [4 5 6])
        result (add a b)]
    (is (equals result (array [5 7 9])))
    (is (equals a (array [1 2 3])))
    (is (equals b (array [4 5 6])))
    (is (= (rank result) 1)))
  (let [result (add (array [1 2 3]) (array 2) (array [4 5 6]))]
    (is (equals result (array [7 9 11])))
    (is (= (rank result) 1)))
  (let [result (add (array 2) (array [1 2 3]) (array [4 5 6]))]
    (is (equals result (array [7 9 11])))
    (is (= (rank result) 1)))
  (let [result (add (array [1 2 3]) (array [4 5 6]) (array 2))]
    (is (equals result (array [7 9 11])))
    (is (= (rank result) 1)))
  (let [result (add (array [[1 2 3] [4 5 6]]) (array [3 2 1]))]
    (is (equals result (array [[4 4 4] [7 7 7]])))
    (is (= (rank result) 2)))
  (let [result (add (array [3 2 1]) (array [[1 2 3] [4 5 6]]))]
    (is (equals result (array [[4 4 4] [7 7 7]])))
    (is (= (rank result) 2)))
  (is (equals (add (array [[1 2 3] [4 5 6]]) (array [[3 2 1]]))
              (array [[4 4 4] [7 7 7]])))
  (is (equals (add (array [[3 2 1]]) (array [[1 2 3] [4 5 6]]))
              (array [[4 4 4] [7 7 7]])))
  (is (equals (add (array [[1 2 3] [4 5 6]]) (array [2 3]))
              (array [[3 4 5] [7 8 9]])))
  (is (equals (add (array [2 3]) (array [[1 2 3] [4 5 6]]))
              (array [[3 4 5] [7 8 9]])))
  (is (equals (add (array [[1 2 3] [4 5 6]]) (array [[2] [3]]))
              (array [[3 4 5] [7 8 9]])))
  (is (equals (add (array [[2] [3]]) (array [[1 2 3] [4 5 6]]))
              (array [[3 4 5] [7 8 9]])))
  (is (equals (add (array [[1 2 3] [4 5 6]]) (array [[3 2 1]]) (array [2 3]))
              (array [[6 6 6] [10 10 10]])))
  (is (equals (add (array [2 3]) (array [[1 2 3] [4 5 6]]) (array [[3 2 1]]))
              (array [[6 6 6] [10 10 10]])))
  (is (equals (add (array [[1 2 3] [4 5 6]]) (array [2 3]) (array [[3 2 1]]))
              (array [[6 6 6] [10 10 10]])))
  (is (thrown? Exception (add (array [2 3]) (array [[3 2 1]]) (array [[1 2 3] [4 5 6]]))))
  (is (equals (add (array [[1 2 3] [4 5 6]]) (array [[3 2 1]]) (array [[2] [3]]))
              (array [[6 6 6] [10 10 10]])))
  (is (equals (add (array [[2] [3]]) (array [[1 2 3] [4 5 6]]) (array [[3 2 1]]))
              (array [[6 6 6] [10 10 10]])))
  (is (equals (add (array [[1 2 3] [4 5 6]]) (array [[2] [3]]) (array [[3 2 1]]))
              (array [[6 6 6] [10 10 10]])))
  (is (equals (add (array [[1 2 3] [4 5 6]]) (array [3 2 1]) (array [[2] [3]]))
              (array [[6 6 6] [10 10 10]])))
  (is (equals (add (array [[2] [3]]) (array [[1 2 3] [4 5 6]]) (array [3 2 1]))
              (array [[6 6 6] [10 10 10]])))
  (is (equals (add (array [[1 2 3] [4 5 6]]) (array [[2] [3]]) (array [3 2 1]))
              (array [[6 6 6] [10 10 10]])))
  (is (equals (add (array [[1 2 3] [4 5 6]]) (array [2 3]) (array [3 3]))
              (array [[6 7 8] [10 11 12]])))
  (is (equals (add (array [3 3]) (array [[1 2 3] [4 5 6]]) (array [2 3]))
              (array [[6 7 8] [10 11 12]])))
  (is (equals (add (array [3 3]) (array [2 3]) (array [[1 2 3] [4 5 6]]))
              (array [[6 7 8] [10 11 12]])))
  (is (equals (add (array [[1 2 3] [4 5 6]]) (array [1 2 3]) (array 2))
              (array [[4 6 8] [7 9 11]])))
  (is (equals (add (array [[1 2 3] [4 5 6]]) (array 2) (array [1 2 3]))
              (array [[4 6 8] [7 9 11]])))
  (is (equals (add (array 2) (array [[1 2 3] [4 5 6]]) (array [1 2 3]))
              (array [[4 6 8] [7 9 11]])))
  (is (equals (add (array 2) (array [1 2 3]) (array [[1 2 3] [4 5 6]]))
              (array [[4 6 8] [7 9 11]])))
  (is (equals (add (array [[1 2 3] [4 5 6]]) (array [[0 2 1] [4 5 3]]))
              (array [[1 4 4] [8 10 9]])))
  (is (equals (add (array [[1 2 3] [4 5 6]]) (array [[0 2 1] [4 5 3]]) (array [1 2 3]))
              (array [[2 6 7] [9 12 12]])))
  (is (equals (add (array [[1 2 3] [4 5 6]]) (array [1 2 3]) (array [[0 2 1] [4 5 3]]))
              (array [[2 6 7] [9 12 12]])))
  (is (equals (add (array [1 2 3]) (array [[1 2 3] [4 5 6]]) (array [[0 2 1] [4 5 3]]))
              (array [[2 6 7] [9 12 12]])))
  (is (equals (add (array 2) (array [1 2 3]) (array [[1 2 3] [4 5 6]]) (array [[0 2 1] [4 5 3]]))
              (array [[4 8 9] [11 14 14]])))
  (is (equals (add (array [1 2 3]) (array 2) (array [[1 2 3] [4 5 6]]) (array [[0 2 1] [4 5 3]]))
              (array [[4 8 9] [11 14 14]])))
  (is (equals (add (array [1 2 3]) (array [[1 2 3] [4 5 6]]) (array 2) (array [[0 2 1] [4 5 3]]))
              (array [[4 8 9] [11 14 14]])))
  (is (equals (add (array [1 2 3]) (array [[1 2 3] [4 5 6]]) (array [[0 2 1] [4 5 3]]) (array 2))
              (array [[4 8 9] [11 14 14]])))
  (is (equals (add (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [1 2 3]))
              (array [[[2 4 6] [5 7 9]] [[8 10 12] [11 13 15]]])))
  (is (equals (add (array [1 2 3])
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]]))
              (array [[[2 4 6] [5 7 9]] [[8 10 12] [11 13 15]]])))
  (is (equals (add (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[1 2 3]]))
              (array [[[2 4 6] [5 7 9]] [[8 10 12] [11 13 15]]])))
  (is (equals (add (array [[1 2 3]])
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]]))
              (array [[[2 4 6] [5 7 9]] [[8 10 12] [11 13 15]]])))
  (is (equals (add (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[[1 2 3]]]))
              (array [[[2 4 6] [5 7 9]] [[8 10 12] [11 13 15]]])))
  (is (equals (add (array [[[1 2 3]]])
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]]))
              (array [[[2 4 6] [5 7 9]] [[8 10 12] [11 13 15]]])))

  (is (equals (add (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[1 2] [3 4]]))
              (array [[[2 3 4] [6 7 8]] [[10 11 12] [14 15 16]]])))
  (is (equals (add (array [[1 2] [3 4]])
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]]))
              (array [[[2 3 4] [6 7 8]] [[10 11 12] [14 15 16]]])))
  (is (equals (add (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[[1] [2]] [[3] [4]]]))
              (array [[[2 3 4] [6 7 8]] [[10 11 12] [14 15 16]]])))
  (is (equals (add (array [[[1] [2]] [[3] [4]]])
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]]))
              (array [[[2 3 4] [6 7 8]] [[10 11 12] [14 15 16]]])))
  (is (equals (add (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[3 2 1] [6 5 4]]))
              (array [[[4 4 4] [10 10 10]] [[10 10 10] [16 16 16]]])))
  (is (equals (add (array [[3 2 1] [6 5 4]])
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]]))
              (array [[[4 4 4] [10 10 10]] [[10 10 10] [16 16 16]]])))
  (is (equals (add (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[[3 2 1] [6 5 4]]]))
              (array [[[4 4 4] [10 10 10]] [[10 10 10] [16 16 16]]])))

  (is (equals (array [[[4 4 4] [10 10 10]] [[10 10 10] [16 16 16]]])
              (add (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[[3 2 1] [6 5 4]]]))))
  (is (equals (add (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[[3 2 1]] [[6 5 4]]]))
              (array [[[4 4 4] [7 7 7]] [[13 13 13] [16 16 16]]])))
  (is (equals (add (array [[[3 2 1]] [[6 5 4]]])
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]]))
              (array [[[4 4 4] [7 7 7]] [[13 13 13] [16 16 16]]])))
  (is (equals (add (array 2)
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[[3 2 1]] [[6 5 4]]]))
              (array [[[6 6 6] [9 9 9]] [[15 15 15] [18 18 18]]])))
  (is (equals (add (array 2)
                   (array [[[3 2 1]] [[6 5 4]]])
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]]))
              (array [[[6 6 6] [9 9 9]] [[15 15 15] [18 18 18]]])))
  (is (equals (add (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[[3 2 1]] [[6 5 4]]])
                   (array [0 1 0])
                   (array 2))
              (array [[[6 7 6] [9 10 9]] [[15 16 15] [18 19 18]]])))
  (is (equals (add (array [0 1 0])
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[[3 2 1]] [[6 5 4]]])
                   (array 2))
              (array [[[6 7 6] [9 10 9]] [[15 16 15] [18 19 18]]])))
  (is (equals (add (array [0 1 0])
                   (array [[[3 2 1]] [[6 5 4]]])
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array 2))
              (array [[[6 7 6] [9 10 9]] [[15 16 15] [18 19 18]]])))
  (is (equals (add (array [0 1 0])
                   (array [[[3 2 1]] [[6 5 4]]])
                   (array 2)
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]]))
              (array [[[6 7 6] [9 10 9]] [[15 16 15] [18 19 18]]])))

  (is (equals (add (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[[3 2 1]] [[6 5 4]]])
                   (array [[1 2] [3 4]]))
              (array [[[5 5 5] [9 9 9]] [[16 16 16] [20 20 20]]])))
  (is (equals (add (array [[[3 2 1]] [[6 5 4]]])
                   (array [[[1 2 3] [4 5 6]]
                           [[7 8 9] [10 11 12]]])
                   (array [[1 2] [3 4]]))
              (array [[[5 5 5] [9 9 9]] [[16 16 16] [20 20 20]]])))
  (let [a (array [[1 2] [3 4]])
        b (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])
        c (array [[[3 2 1]] [[6 5 4]]])]
    (is (equals (add a b c) (array [[[5 5 5] [9 9 9]] [[16 16 16] [20 20 20]]])))
    (is (equals a (array [[1 2] [3 4]])))
    (is (equals b (array [[[1 2 3] [4 5 6]]
                          [[7 8 9] [10 11 12]]])))
    (is (equals c (array [[[3 2 1]] [[6 5 4]]])))))

(deftest sub!-test
  (is (thrown? Exception (sub! (array 2) (array 3))))
  (is (thrown? Exception (sub! (array 2))))
  (let [a (array [1 2 3])]
    (sub! a)
    (is (equals a (array [-1 -2 -3]))))
  (let [a (array [1 2 3])
        b (array [4 5 6])
        result (sub! a b)]
    (is (equals result (array [-3 -3 -3])))
    (is (equals a result))
    (is (equals b (array [4 5 6])))
    (is (= (rank result) 1)))
  (let [a (array [[1 2 3] [4 5 6]])
        b (array [[3 2 1]])
        c (array [2 3])]
    (sub! a b c)
    (is (equals a (array [[-4 -2 0] [-2 0 2]])))
    (is (equals b (array [[3 2 1]])))
    (is (equals c (array [2 3])))
    (is (thrown? Exception (sub! c b a))))
  (let [a (array [[1 2 3] [4 5 6]])
        b (array [[0 2 1] [4 5 3]])]
    (sub! a b)
    (is (equals a (array [[1 0 2] [0 0 3]])))
    (is (equals b (array [[0 2 1] [4 5 3]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])]
    (sub! a (array [[[3 2 1]] [[6 5 4]]]) (array [0 1 0]) (array 2))
    (is (equals a (array [[[-4 -3 0] [-1 0 3]] [[-1 0 3] [2 3 6]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])
        b (array [[1 2] [3 4]])
        c (array [[[3 2 1]] [[6 5 4]]])
        result (sub! a b c)]
    (is (equals result (array [[[-3 -1 1] [-1 1 3]] [[-2 0 2] [0 2 4]]])))
    (is (equals a result))
    (is (equals b (array [[1 2] [3 4]])))
    (is (equals c (array [[[3 2 1]] [[6 5 4]]])))))

(deftest sub-test
  (is (equals (sub) (array 0)))
  (is (equals (sub (array [1 2 3])) (array [-1 -2 -3])))
  (is (equals (sub (array 7)) (array -7)))
  (is (equals (sub (array 3) (array 2)) (array 1)))
  (let [a (array [1 2 3])
        b (array [4 5 6])
        result (sub a b)]
    (is (equals result (array [-3 -3 -3])))
    (is (equals a (array [1 2 3])))
    (is (equals b (array [4 5 6])))
    (is (= (rank result) 1)))
  (let [a (array [[1 2 3] [4 5 6]])
        b (array [[3 2 1]])
        c (array [2 3])
        result (sub a b c)]
    (is (equals result (array [[-4 -2 0] [-2 0 2]])))
    (is (equals a (array [[1 2 3] [4 5 6]])))
    (is (equals b (array [[3 2 1]])))
    (is (equals c (array [2 3])))
    (is (thrown? Exception (sub c b a))))
  (is (equals (sub (array [[1 2 3] [4 5 6]])
                   (array [[0 2 1] [4 5 3]])) (array [[1 0 2] [0 0 3]])))
  (is (equals (sub (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]])
                   (array [[[3 2 1]] [[6 5 4]]])
                   (array [0 1 0])
                   (array 2))
              (array [[[-4 -3 0] [-1 0 3]] [[-1 0 3] [2 3 6]]])))
  (is (equals (sub (array [[[3 2 1]] [[6 5 4]]])
                   (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]])
                   (array 2)
                   (array [0 1 0]))
              (array [[[0 -3 -4] [-3 -6 -7]] [[-3 -6 -7] [-6 -9 -10]]])))
  (is (equals (sub (array [[[3 2 1]] [[6 5 4]]])
                   (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]])
                   (array [0 1 0])
                   (array 2))
              (array [[[0 -3 -4] [-3 -6 -7]] [[-3 -6 -7] [-6 -9 -10]]])))
  (is (equals (sub (array [0 1 0])
                   (array [[[3 2 1]] [[6 5 4]]])
                   (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]])
                   (array 2))
              (array [[[-6 -5 -6] [-9 -8 -9]] [[-15 -14 -15] [-18 -17 -18]]])))
  (is (equals (sub (array 2)
                   (array [0 1 0])
                   (array [[[3 2 1]] [[6 5 4]]])
                   (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]]))
              (array [[[-2 -3 -2] [-5 -6 -5]] [[-11 -12 -11] [-14 -15 -14]]])))
  (is (equals (sub (array 2)
                   (array [[[3 2 1]] [[6 5 4]]])
                   (array [0 1 0])
                   (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]]))
              (array [[[-2 -3 -2] [-5 -6 -5]] [[-11 -12 -11] [-14 -15 -14]]])))
  (is (equals (sub (array 2)
                   (array [[[1 2 3] [4 5 6]] [[7 8 9] [10 11 12]]])
                   (array [[[3 2 1]] [[6 5 4]]])
                   (array [0 1 0]))
              (array [[[-2 -3 -2] [-5 -6 -5]] [[-11 -12 -11] [-14 -15 -14]]])))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[7 8 9] [10 11 12]]])
        b (array [[1 2] [3 4]])
        c (array [[[3 2 1]] [[6 5 4]]])
        result (sub a b c)]
    (is (equals result (array [[[-3 -1 1] [-1 1 3]] [[-2 0 2] [0 2 4]]])))
    (is (equals a (array [[[1 2 3] [4 5 6]]
                          [[7 8 9] [10 11 12]]])))
    (is (equals b (array [[1 2] [3 4]])))
    (is (equals c (array [[[3 2 1]] [[6 5 4]]])))))

(deftest div!-test
  (is (thrown? Exception (div! (array 2) (array 3))))
  (is (thrown? Exception (div! (array 2))))
  (let [a (array [1 2 4])
        result (div! a)]
    (is (equals a (array [1 0.5 0.25])))
    (is (equals result a)))
  (let [a (array [4 5 6])
        b (array [1 2 3])
        result (div! a b)]
    (is (equals result (array [4 2.5 2])))
    (is (equals a result))
    (is (equals b (array [1 2 3])))
    (is (= (rank result) 1)))
  (let [a (array [[2 0 -3] [4 5 6]])
        b (array [[1 2 3]])
        c (array [2 4])]
    (div! a b c)
    (is (equals a (array [[1 0 -0.5] [1 0.625 0.5]])))
    (is (equals b (array [[1 2 3]])))
    (is (equals c (array [2 4])))
    (is (thrown? Exception (div! c b a))))
  (let [a (array [[0 2 3] [2 7.5 6]])
        b (array [[1 2 1] [4 5 3]])]
    (div! a b)
    (is (equals a (array [[0 1 3] [0.5 1.5 2]])))
    (is (equals b (array [[1 2 1] [4 5 3]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[9 8 9] [10 2.5 12]]])]
    (div! a (array [[[4 2 1]] [[2.5 5 4]]]) (array [1 -1 2]) (array 2))
    (is (equals a (array [[[0.125 -0.5 0.75] [0.5 -1.25 1.5]] [[1.8 -0.8 0.5625] [2 -0.25 0.75]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[6 3 12] [10 8 9]]])
        b (array [[1 2] [3 4]])
        c (array [[[4 2 1]] [[5 5 4]]])
        result (div! a b c)]
    (is (equals result (array [[[0.25 1 3] [0.5 1.25 3]] [[0.4 0.2 1] [0.5 0.4 0.5625]]])))
    (is (equals a result))
    (is (equals b (array [[1 2] [3 4]])))
    (is (equals c (array [[[4 2 1]] [[5 5 4]]])))))

(deftest div-test
  (is (equals (div (array 2)) (array 0.5)))
  (let [a (array [1 2 4])
        result (div a)]
    (is (equals result (array [1 0.5 0.25])))
    (is (equals a (array [1 2 4]))))
  (let [a (array [4 5 6])
        b (array [1 2 3])
        result (div a b)]
    (is (equals result (array [4 2.5 2])))
    (is (equals a (array [4 5 6])))
    (is (equals b (array [1 2 3])))
    (is (= (rank result) 1)))
  (let [a (array [[2 0 -3] [4 5 6]])
        b (array [[1 2 3]])
        c (array [2 4])]
    (is (equals (div a b c) (array [[1 0 -0.5] [1 0.625 0.5]])))
    (is (thrown? Exception (div c b a))))
  (is (equals (div (array [[0 2 3] [2 7.5 6]]) (array [[1 2 1] [4 5 3]]))
              (array [[0 1 3] [0.5 1.5 2]])))
  (is (equals (div (array [[[1 2 3] [4 5 6]] [[9 8 9] [10 2.5 12]]])
                   (array [[[4 2 1]] [[2.5 5 4]]])
                   (array [1 -1 2])
                   (array 2))
              (array [[[0.125 -0.5 0.75] [0.5 -1.25 1.5]] [[1.8 -0.8 0.5625] [2 -0.25 0.75]]])))
  (is (equals (div (array [[[1 2 3] [4 5 6]] [[9 8 9] [10 2.5 12]]])
                   (array [[[4 2 1]] [[2.5 5 4]]])
                   (array 2)
                   (array [1 -1 2]))
              (array [[[0.125 -0.5 0.75] [0.5 -1.25 1.5]] [[1.8 -0.8 0.5625] [2 -0.25 0.75]]])))
  (is (equals (div (array [[[1 2 3] [4 5 6]] [[9 8 9] [10 2.5 12]]])
                   (array 2)
                   (array [[[4 2 1]] [[2.5 5 4]]])
                   (array [1 -1 2]))
              (array [[[0.125 -0.5 0.75] [0.5 -1.25 1.5]] [[1.8 -0.8 0.5625] [2 -0.25 0.75]]])))
  (is (equals (div (array 2.4)
                   (array [[[1 2 4] [4 5 6]] [[3 8 1.2] [10 2.4 12]]])
                   (array [[[4 2 1]] [[2.5 5 4]]])
                   (array [1 -1 2]))
              (array [[[0.6 -0.6 0.3] [0.15 -0.24 0.2]] [[0.32 -0.06 0.25] [0.096 -0.2 0.025]]])))
  (is (equals (div (array 2.4)
                   (array [1 -1 2])
                   (array [[[1 2 4] [4 5 6]] [[3 8 1.2] [10 2.4 12]]])
                   (array [[[4 2 1]] [[2.5 5 4]]]))
              (array [[[0.6 -0.6 0.3] [0.15 -0.24 0.2]] [[0.32 -0.06 0.25] [0.096 -0.2 0.025]]])))
  (is (equals (div (array 2.4)
                   (array [[[4 2 1]] [[2.5 5 4]]])
                   (array [1 -1 2])
                   (array [[[1 2 4] [4 5 6]] [[3 8 1.2] [10 2.4 12]]]))
              (array [[[0.6 -0.6 0.3] [0.15 -0.24 0.2]] [[0.32 -0.06 0.25] [0.096 -0.2 0.025]]])))
  (let [a (array [[[1 2 3] [4 5 6]] [[6 3 12] [10 8 9]]])
        b (array [[1 2] [3 4]])
        c (array [[[4 2 1]] [[5 5 4]]])
        result (div a b c)]
    (is (equals result (array [[[0.25 1 3] [0.5 1.25 3]] [[0.4 0.2 1] [0.5 0.4 0.5625]]])))
    (is (equals a (array [[[1 2 3] [4 5 6]] [[6 3 12] [10 8 9]]])))
    (is (equals b (array [[1 2] [3 4]])))
    (is (equals c (array [[[4 2 1]] [[5 5 4]]])))))

(deftest pow!-test
  (is (equals (pow! (array 7)) (array 7)))
  (let [a (array 2)
        b (array 3)
        result (pow! a b)]
    (is (equals a (array 8)))
    (is (equals result a))
    (is (equals b (array 3))))
  (let [a (array [1 2 4])
        b (array 2)
        result (pow! a b)]
    (is (equals a (array [1 4 16])))
    (is (equals result a))
    (is (equals b (array 2))))
  (is (thrown? Exception (pow! (array 2) (array [1 2 4]))))
  (let [a (array [4 5 6])
        b (array [3 2 1])
        result (pow! a b)]
    (is (equals result (array [64 25 6])))
    (is (equals a result))
    (is (equals b (array [3 2 1]))))
  (is (equals (pow! (array [4 5 6]) (array [3 2 1]) (array 2)) (array [4096 625 36])))
  (is (equals (pow! (array [4 5 6]) (array 2) (array [3 2 1])) (array [4096 625 36])))
  (let [a (array [[2 0 -3] [4 5 6]])
        b (array [[1 2 3]])
        c (array [2 3])]
    (pow! a b c)
    (is (equals a (array [[4 0 729] [64 15625 10077696]])))
    (is (equals b (array [[1 2 3]])))
    (is (equals c (array [2 3])))
    (is (thrown? Exception (pow! c b a))))
  (let [a (array [[0 2 3] [2 7 3]])
        b (array [[4 2 1] [0 2 3]])]
    (pow! a b)
    (is (equals a (array [[0 4 3] [1 49 27]])))
    (is (equals b (array [[4 2 1] [0 2 3]]))))
  (is (equals (pow! (array [[[1 -2 3] [4 5 6]]
                            [[0 4 6] [1 2.5 -0.3]]]) (array [[[0 1 2]] [[3 2 2]]]))
              (array [[[1 -2 9] [1 5 36]] [[0 16 36] [1 6.25 0.09]]])))
  (is (equals (pow! (array [[[1 -2 3] [4 5 6]]
                            [[0 4 6] [1 2.5 -0.3]]]) (array [[[0 1 2]] [[3 2 2]]]) (array [2 2 1]))
              (array [[[1 4 9] [1 25 36]] [[0 256 36] [1 39.0625 0.09]]])))
  (let [a (array [[[1 -2 3] [4 5 6]]
                  [[0 4 6] [1 2.5 -0.3]]])]
    (pow! a (array [[[0 1 2]] [[3 2 2]]]) (array [2 2 1]) (array 2))
    (is (equals a (array [[[1 16 81] [1 625 1296]] [[0 65536 1296] [1 1525.87890625 0.0081]]]))))
  (let [a (array [[[1 2 3] [4 5 6]]
                  [[6 3 12] [10 8 9]]])
        b (array [[3 2] [1 0]])
        c (array [[[0 2 1]] [[2 1 2]]])
        result (pow! a b c)]
    (is (equals result (array [[[1 64 27] [1 625 36]] [[36 3 144] [1 1 1]]])))
    (is (equals a result))
    (is (equals b (array [[3 2] [1 0]])))
    (is (equals c (array [[[0 2 1]] [[2 1 2]]])))))

(deftest pow-test
  (is (equals (pow (array 7)) (array 7)))
  (let [a (array 2)
        b (array 3)
        result (pow a b)]
    (is (equals result (array 8)))
    (is (equals a (array 2)))
    (is (equals b (array 3))))
  (let [a (array [1 2 4])
        b (array 2)
        result (pow a b)]
    (is (equals result (array [1 4 16])))
    (is (equals a (array [1 2 4])))
    (is (equals b (array 2))))
  (is (equals (pow (array 2) (array [1 2 4])) (array [2 4 16])))
  (is (equals (pow (array [4 5 6]) (array [3 2 1])) (array [64 25 6])))
  (is (equals (pow (array [4 5 6]) (array [3 2 1]) (array 2)) (array [4096 625 36])))
  (is (equals (pow (array [4 5 6]) (array 2) (array [3 2 1])) (array [4096 625 36])))
  (is (equals (pow (array 2) (array [4 5 6]) (array [3 2 1])) (array [4096 1024 64])))
  (is (equals (pow (array 2) (array 3) (array [3 2 1])) (array [512 64 8])))
  (let [a (array [[2 0 -3] [4 5 6]])
        b (array [[1 2 3]])
        c (array [2 3])
        result (pow a b c)]
    (is (equals result (array [[4 0 729] [64 15625 10077696]])))
    (is (equals a (array [[2 0 -3] [4 5 6]])))
    (is (equals b (array [[1 2 3]])))
    (is (equals c (array [2 3]))))
  (let [a (array [2 3])
        b (array [[2 0 3] [4 2 1]])
        c (array [[1 2 3]])
        result (pow a b c)]
    (is (equals result (array [[4 1 512] [81 81 27]])))
    (is (equals a (array [2 3])))
    (is (equals b (array [[2 0 3] [4 2 1]])))
    (is (equals c (array [[1 2 3]]))))
  (is (equals (pow (array [[0 2 3] [2 7 3]]) (array [[4 2 1] [0 2 3]]))
              (array [[0 4 3] [1 49 27]])))
  (is (equals (pow (array [[[1 -2 3] [4 5 6]]
                           [[0 4 6] [1 2.5 -0.3]]]) (array [[[0 1 2]] [[3 2 2]]]))
              (array [[[1 -2 9] [1 5 36]] [[0 16 36] [1 6.25 0.09]]])))
  (is (equals (pow (array [[[1 -2 3] [4 5 6]]
                           [[0 4 6] [1 2.5 -0.3]]]) (array [[[0 1 2]] [[3 2 2]]]) (array [2 2 1]))
              (array [[[1 4 9] [1 25 36]] [[0 256 36] [1 39.0625 0.09]]])))
  (is (equals (pow (array [[[1 -2 3] [4 5 6]] [[0 4 6] [1 2.5 -0.3]]])
                   (array [[[0 1 2]] [[3 2 2]]])
                   (array [2 2 1])
                   (array 2))
              (array [[[1 16 81] [1 625 1296]] [[0 65536 1296] [1 1525.87890625 0.0081]]])))
  (is (equals (pow (array [[[1 -2 3] [4 5 6]] [[0 4 6] [1 2.5 -0.3]]])
                   (array [[[0 1 2]] [[3 2 2]]])
                   (array 2)
                   (array [2 2 1]))
              (array [[[1 16 81] [1 625 1296]] [[0 65536 1296] [1 1525.87890625 0.0081]]])))
  (is (equals (pow (array [[[1 -2 3] [4 5 6]] [[0 4 6] [1 2.5 -0.3]]])
                   (array 2)
                   (array [[[0 1 2]] [[3 2 2]]])
                   (array [2 2 1]))
              (array [[[1 16 81] [1 625 1296]] [[0 65536 1296] [1 1525.87890625 0.0081]]])))
  (is (equals (pow (array [[[1 -2 3] [4 5 6]] [[0 4 6] [1 2.5 -0.3]]])
                   (array 2)
                   (array [2 2 1])
                   (array [[[0 1 2]] [[3 2 2]]]))
              (array [[[1 16 81] [1 625 1296]] [[0 65536 1296] [1 1525.87890625 0.0081]]])))
  (is (equals (pow (array [[[1 2 3] [4 5 6]] [[6 3 12] [10 8 9]]])
                   (array [[3 2] [1 0]])
                   (array [[[0 2 1]] [[2 1 2]]]))
              (array [[[1 64 27] [1 625 36]] [[36 3 144] [1 1 1]]]))))

(deftest mmul-test
  (testing "no arguments"
    (is (equals (mmul) (array 1))))
  (testing "one argument"
    (is (equals (mmul (array 2)) (array 2)))
    (is (equals (mmul (array [1 2 3])) (array [1 2 3])))
    (is (equals (mmul (array [[1 2] [3 4]])) (array [[1 2] [3 4]])))
    (is (equals (mmul (array [[[1 2] [3 4]] [[1 2] [3 4]]])) (array [[[1 2] [3 4]] [[1 2] [3 4]]]))))
  (testing "only scalar arguments"
    (is (equals (mmul (array 2) (array 3)) (array 6)))
    (is (equals (mmul (array 2) (array 3) (array 4) (array 5)) (array 120))))
  (testing "vector * scalar"
    (is (equals (mmul (array [1 2 3]) (array 2)) (array [2 4 6])))
    (is (equals (mmul (array 2) (array [1 2 3])) (array [2 4 6])))
    (is (equals (mmul (array [1 2 3]) (array 2) (array 3)) (array [6 12 18])))
    (is (equals (mmul (array 3) (array [1 2 3]) (array 2)) (array [6 12 18])))
    (is (equals (mmul (array 2) (array 3) (array [1 2 3])) (array [6 12 18]))))
  (testing "matrix * scalar"
    (is (equals (mmul (array [[1 2] [3 4]]) (array 2)) (array [[2 4] [6 8]])))
    (is (equals (mmul (array 2) (array [[1 2] [3 4]])) (array [[2 4] [6 8]])))
    (is (equals (mmul (array [[1 2] [3 4]]) (array 2) (array 3)) (array [[6 12] [18 24]])))
    (is (equals (mmul (array 3) (array [[1 2] [3 4]]) (array 2)) (array [[6 12] [18 24]])))
    (is (equals (mmul (array 3) (array 2) (array [[1 2] [3 4]])) (array [[6 12] [18 24]]))))
  (testing "tensor * scalar"
    (is (equals (mmul (array [[[1 2] [3 4]] [[5 6] [7 8]]]) (array 2))
                (array [[[2 4] [6 8]] [[10 12] [14 16]]])))
    (is (equals (mmul (array 2) (array [[[1 2] [3 4]] [[5 6] [7 8]]]))
                (array [[[2 4] [6 8]] [[10 12] [14 16]]])))
    (is (equals (mmul (array [[[1 2] [3 4]] [[5 6] [7 8]]]) (array 2) (array 3))
                (array [[[6 12] [18 24]] [[30 36] [42 48]]])))
    (is (equals (mmul (array 3) (array [[[1 2] [3 4]] [[5 6] [7 8]]]) (array 2))
                (array [[[6 12] [18 24]] [[30 36] [42 48]]])))
    (is (equals (mmul (array 3) (array 2) (array [[[1 2] [3 4]] [[5 6] [7 8]]]))
                (array [[[6 12] [18 24]] [[30 36] [42 48]]]))))
  (testing "vector * vector"
    (is (thrown? Exception (mmul (array [1 2]) (array [4 5 6]))))
    (is (thrown? Exception (mmul (array [1 2 3]) (array [4 5]))))
    (is (equals (mmul (array [1 2 3]) (array [4 5 6])) (array 32))))
  (testing "vector * vector * vector"
    (is (equals (mmul (array [1 2 3]) (array [4 5 6]) (array [0 1 2])) (array [0 32 64])))
    (is (equals (mmul (array [1 2 3]) (array [0 1 2]) (array [4 5 6])) (array [32 40 48])))
    (is (equals (mmul (array [1 2 3]) (array [4 5 6]) (array [1 2])) (array [32 64]))))
  (testing "vector * vector * scalar"
    (is (equals (mmul (array [1 2 3]) (array [4 5 6]) (array 2)) (array 64)))
    (is (equals (mmul (array [1 2 3]) (array 2) (array [4 5 6])) (array 64)))
    (is (equals (mmul (array 2) (array [1 2 3]) (array [4 5 6])) (array 64))))
  (testing "matrix * vector"
    (is (equals (mmul (array [[1 2 3] [4 5 6]]) (array [2 1 3])) (array [13 31])))
    (is (equals (mmul (array [2 1 3]) (array [[1 4] [2 5] [3 6]])) (array [13 31]))))
  (testing "matrix * vector * scalar"
    (is (equals (mmul (array [[1 2 3] [4 5 6]]) (array [2 1 3]) (array 2)) (array [26 62])))
    (is (equals (mmul (array [[1 2 3] [4 5 6]]) (array 2) (array [2 1 3])) (array [26 62])))
    (is (equals (mmul (array 2) (array [[1 2 3] [4 5 6]]) (array [2 1 3])) (array [26 62]))))
  (testing "matrix * vector * vector"
    (is (equals (mmul (array [[1 2 3] [4 5 6]]) (array [2 1 3]) (array [2 1])) (array 57)))
    (is (equals (mmul (array [2 1 3]) (array [[1 4] [2 5] [3 6]]) (array [2 1])) (array 57))))
  (testing "matrix * matrix"
    (is (equals (mmul (array [[1 2 3] [4 5 6]]) (array [[0 2] [3 1] [2 4]])) (array [[12 16] [27 37]])))
    (is (thrown? Exception (mmul (array [[1 2 3] [4 5 6]]) (array [[1 2 3] [4 5 6]]))))
    (is (equals (mmul (array [[0 2] [3 1] [2 4]]) (array [[1 3 5] [2 4 6]]))
                (array [[4 8 12] [5 13 21] [10 22 34]])))
    (is (thrown? Exception (mmul (array [[1 2 3]]) (array [[1 2 3]]))))
    (is (equals (mmul (array [[1 2 3]]) (array [[1] [2] [3]])) (array [[14]])))
    (is (equals (mmul (array [[1] [2] [3]]) (array [[1 2 3]])) (array [[1 2 3] [2 4 6] [3 6 9]]))))
  (testing "matrix * matrix * matrix"
    (is (equals (mmul (array [[1 2 3] [4 5 6]]) (array [[0 2] [3 1] [2 4]]) (array [[0 1] [2 1]]))
                (array [[32 28] [74 64]]))))
  (testing "matrix * vector * matrix"
    (is (equals (mmul (array [[1 2 3] [4 5 6]]) (array [2 1 3]) (array [[1 3] [2 0]]))
                (array [75 39]))))
  (testing "matrix * tensor"
    (is (equals (mmul (array [[2 1 3] [1 1 2]]) (array [[[1 2] [4 6]]
                                                        [[2 3] [5 5]]
                                                        [[3 1] [6 4]]]))
                (array [[[13 10] [31 29]] [[9 7] [21 19]]])))
    (is (equals (mmul (array [[[1 2 3]
                               [4 5 6]]
                              [[2 3 1]
                               [6 5 4]]]) (array [[2 1]
                                                  [1 1]
                                                  [3 2]]))
                (array [[[13 9] [31 21]] [[10 7] [29 19]]]))))
  (testing "tensor * tensor"
    (is (equals (mmul (array [[[1 0 1]
                               [2 1 3]]
                              [[0 2 0]
                               [3 1 1]]]) (array [[[1 2] [2 1]]
                                                  [[1 1] [1 2]]
                                                  [[2 1] [1 1]]]))
                (array [[[[3 3] [3 2]] [[9 8] [8 7]]]
                        [[[2 2] [2 4]] [[6 8] [8 6]]]]))))
  (testing "tensor * tensor * tensor"
    (is (equals (mmul (array [[[1 0 1]
                               [2 1 3]]
                              [[0 2 0]
                               [3 1 1]]]) (array [[[1 2] [2 1]]
                                                  [[1 1] [1 2]]
                                                  [[2 1] [1 1]]]) (array [[[1] [2] [3]] [[4] [5] [6]]]))
                (array [[[[[15] [21] [27]] [[11] [16] [21]]] [[[41] [58] [75]] [[36] [51] [66]]]]
                        [[[[10] [14] [18]] [[18] [24] [30]]] [[[38] [52] [66]] [[32] [46] [60]]]]]))))
  (testing "tensor * vector"
    (is (equals (mmul (array [[[1 0 1]
                               [2 1 3]]
                              [[0 2 0]
                               [3 1 1]]]) (array [1 2 3]))
                (array [[4 13] [4 8]])))
    (is (equals (mmul (array [1 2]) (array [[[1 0 1]
                                             [2 1 3]]
                                            [[0 2 0]
                                             [3 1 1]]]))
                (array [[1 4 1] [8 3 5]]))))
  (testing "scalar * vector * tensor"
    (is (equals (mmul (array 2) (array [[[1 0 1]
                                         [2 1 3]]
                                        [[0 2 0]
                                         [3 1 1]]]) (array [1 2 3]))
                (array [[8 26] [8 16]])))
    (is (equals (mmul (array [[[1 0 1]
                               [2 1 3]]
                              [[0 2 0]
                               [3 1 1]]]) (array 2) (array [1 2 3]))
                (array [[8 26] [8 16]])))))

(deftest outer-product-test
  (testing "no arguments"
    (is (equals (outer-product) (array 1))))
  (testing "one argument"
    (is (equals (outer-product (array 2)) (array 2)))
    (is (equals (outer-product (array [1 2 3])) (array [1 2 3])))
    (is (equals (outer-product (array [[1 2] [3 4]])) (array [[1 2] [3 4]])))
    (is (equals (outer-product (array [[[1 2] [3 4]] [[1 2] [3 4]]])) (array [[[1 2] [3 4]] [[1 2] [3 4]]]))))
  (testing "only scalar arguments"
    (is (equals (outer-product (array 2) (array 3)) (array 6)))
    (is (equals (outer-product (array 2) (array 3) (array 4) (array 5)) (array 120))))
  (testing "vector * scalar"
    (is (equals (outer-product (array [1 2 3]) (array 2)) (array [2 4 6])))
    (is (equals (outer-product (array 2) (array [1 2 3])) (array [2 4 6])))
    (is (equals (outer-product (array [1 2 3]) (array 2) (array 3)) (array [6 12 18])))
    (is (equals (outer-product (array 3) (array [1 2 3]) (array 2)) (array [6 12 18])))
    (is (equals (outer-product (array 2) (array 3) (array [1 2 3])) (array [6 12 18]))))
  (testing "matrix * scalar"
    (is (equals (outer-product (array [[1 2] [3 4]]) (array 2)) (array [[2 4] [6 8]])))
    (is (equals (outer-product (array 2) (array [[1 2] [3 4]])) (array [[2 4] [6 8]])))
    (is (equals (outer-product (array [[1 2] [3 4]]) (array 2) (array 3)) (array [[6 12] [18 24]])))
    (is (equals (outer-product (array 3) (array [[1 2] [3 4]]) (array 2)) (array [[6 12] [18 24]])))
    (is (equals (outer-product (array 3) (array 2) (array [[1 2] [3 4]])) (array [[6 12] [18 24]]))))
  (testing "tensor * scalar"
    (is (equals (outer-product (array [[[1 2] [3 4]] [[5 6] [7 8]]]) (array 2))
                (array [[[2 4] [6 8]] [[10 12] [14 16]]])))
    (is (equals (outer-product (array 2) (array [[[1 2] [3 4]] [[5 6] [7 8]]]))
                (array [[[2 4] [6 8]] [[10 12] [14 16]]])))
    (is (equals (outer-product (array [[[1 2] [3 4]] [[5 6] [7 8]]]) (array 2) (array 3))
                (array [[[6 12] [18 24]] [[30 36] [42 48]]])))
    (is (equals (outer-product (array 3) (array [[[1 2] [3 4]] [[5 6] [7 8]]]) (array 2))
                (array [[[6 12] [18 24]] [[30 36] [42 48]]])))
    (is (equals (outer-product (array 3) (array 2) (array [[[1 2] [3 4]] [[5 6] [7 8]]]))
                (array [[[6 12] [18 24]] [[30 36] [42 48]]]))))
  (testing "vector * vector"
    (is (equals (outer-product (array [1 2 3]) (array [4 5 6])) (array [[4 5 6] [8 10 12] [12 15 18]])))
    (is (equals (outer-product (array [4 5]) (array [1 2 3])) (array [[4 8 12] [5 10 15]])))
    (is (equals (outer-product (array [1 2]) (array [4 5])) (array [[4 5] [8 10]]))))
  (testing "vector * vector * scalar"
    (is (equals (outer-product (array 2) (array [4 5]) (array [1 2 3])) (array [[8 16 24] [10 20 30]])))
    (is (equals (outer-product (array [4 5]) (array 2) (array [1 2 3])) (array [[8 16 24] [10 20 30]])))
    (is (equals (outer-product (array [4 5]) (array [1 2 3]) (array 2)) (array [[8 16 24] [10 20 30]])))
    (is (equals (outer-product (array 2) (array [1 2 3]) (array [4 5])) (array [[8 10] [16 20] [24 30]])))
    (is (equals (outer-product (array [1 2 3]) (array 2) (array [4 5])) (array [[8 10] [16 20] [24 30]])))
    (is (equals (outer-product (array [1 2 3]) (array [4 5]) (array 2)) (array [[8 10] [16 20] [24 30]]))))
  (testing "matrix * vector"
    (is (equals (outer-product (array [[4 8 12] [5 10 15]]) (array [1 2]))
                (array [[[4 8] [8 16] [12 24]] [[5 10] [10 20] [15 30]]])))
    (is (equals (outer-product (array [1 2]) (array [[4 8 12] [5 10 15]]))
                (array [[[4 8 12] [5 10 15]] [[8 16 24] [10 20 30]]])))
    (is (equals (outer-product (array [[4 8 12] [5 10 15]]) (array [3 4 5]))
                (array [[[12 16 20] [24 32 40] [36 48 60]] [[15 20 25] [30 40 50] [45 60 75]]])))
    (is (equals (outer-product (array [3 4 5]) (array [[4 8 12] [5 10 15]]))
                (array [[[12 24 36] [15 30 45]] [[16 32 48] [20 40 60]] [[20 40 60] [25 50 75]]]))))
  (testing "matrix * vector * scalar"
    (is (equals (outer-product (array 2) (array [[4 8 12] [5 10 15]]) (array [1 2]))
                (array [[[8 16] [16 32] [24 48]] [[10 20] [20 40] [30 60]]])))
    (is (equals (outer-product (array [1 2]) (array 2) (array [[4 8 12] [5 10 15]]))
                (array [[[8 16 24] [10 20 30]] [[16 32 48] [20 40 60]]])))
    (is (equals (outer-product (array [[4 8 12] [5 10 15]]) (array [3 4 5]) (array 2))
                (array [[[24 32 40] [48 64 80] [72 96 120]] [[30 40 50] [60 80 100] [90 120 150]]]))))
  (testing "vector * vector * vector"
    (is (equals (outer-product (array [4 5]) (array [1 2 3]) (array [1 2]))
                (array [[[4 8] [8 16] [12 24]] [[5 10] [10 20] [15 30]]])))
    (is (equals (outer-product (array [1 2]) (array [4 5]) (array [1 2 3]))
                (array [[[4 8 12] [5 10 15]] [[8 16 24] [10 20 30]]])))
    (is (equals (outer-product (array [4 5]) (array [1 2 3]) (array [3 4 5]))
                (array [[[12 16 20] [24 32 40] [36 48 60]] [[15 20 25] [30 40 50] [45 60 75]]])))
    (is (equals (outer-product (array [3 4 5]) (array [4 5]) (array [1 2 3]))
                (array [[[12 24 36] [15 30 45]] [[16 32 48] [20 40 60]] [[20 40 60] [25 50 75]]]))))
  (testing "vector * vector * vector * scalar"
    (is (equals (outer-product (array [4 5]) (array [1 2 3]) (array [1 2]) (array 2))
                (array [[[8 16] [16 32] [24 48]] [[10 20] [20 40] [30 60]]])))
    (is (equals (outer-product (array [1 2]) (array [4 5]) (array 2) (array [1 2 3]))
                (array [[[8 16 24] [10 20 30]] [[16 32 48] [20 40 60]]])))
    (is (equals (outer-product (array [4 5]) (array 2) (array [1 2 3]) (array [3 4 5]))
                (array [[[24 32 40] [48 64 80] [72 96 120]] [[30 40 50] [60 80 100] [90 120 150]]])))
    (is (equals (outer-product (array 2) (array [3 4 5]) (array [4 5]) (array [1 2 3]))
                (array [[[24 48 72] [30 60 90]] [[32 64 96] [40 80 120]] [[40 80 120] [50 100 150]]]))))
  (testing "matrix * matrix"
    (is (equals (outer-product (array [[1 2]]) (array [[3 4]])) (array [[[[3 4]] [[6 8]]]])))
    (is (equals (outer-product (array [[1 2]]) (array [[3] [4]])) (array [[[[3] [4]] [[6] [8]]]])))
    (is (equals (outer-product (array [[1] [2]]) (array [[3 4]])) (array [[[[3 4]]] [[[6 8]]]])))
    (is (equals (outer-product (array [[1] [2]]) (array [[3] [4]])) (array [[[[3] [4]]] [[[6] [8]]]])))
    (is (equals (outer-product (array [[1 2] [0 -1]]) (array [[3 4]])) (array [[[[3 4]] [[6 8]]] [[[0 0]] [[-3 -4]]]])))
    (is (equals (outer-product (array [[1 2] [0 -1]]) (array [[3 4]])) (array [[[[3 4]] [[6 8]]] [[[0 0]] [[-3 -4]]]])))
    (is (equals (outer-product (array [[3 4]]) (array [[1 2] [0 -1]])) (array [[[[3 6] [0 -3]] [[4 8] [0 -4]]]])))
    (is (equals (outer-product (array [[1 2] [0 -1]]) (array [[1] [2]])) (array [[[[1] [2]] [[2] [4]]] [[[0] [0]] [[-1] [-2]]]])))
    (is (equals (outer-product (array [[1] [2]]) (array [[1 2] [0 -1]])) (array [[[[1 2] [0 -1]]] [[[2 4] [0 -2]]]])))
    (is (equals (outer-product (array [[3 4] [5 6]]) (array [[1 2] [0 -1]]))
                (array [[[[3 6] [0 -3]] [[4 8] [0 -4]]] [[[5 10] [0 -5]] [[6 12] [0 -6]]]]))))
  (testing "matrix * matrix * scalar"
    (is (equals (outer-product (array [[1 2]]) (array [[3 4]]) (array 2)) (array [[[[6 8]] [[12 16]]]])))
    (is (equals (outer-product (array [[1 2]]) (array 2) (array [[3] [4]])) (array [[[[6] [8]] [[12] [16]]]])))
    (is (equals (outer-product (array [[1 2]]) (array [[3] [4]]) (array 2)) (array [[[[6] [8]] [[12] [16]]]])))
    (is (equals (outer-product (array 2) (array [[1] [2]]) (array [[3 4]])) (array [[[[6 8]]] [[[12 16]]]])))
    (is (equals (outer-product (array [[1] [2]]) (array [[3 4]]) (array 2)) (array [[[[6 8]]] [[[12 16]]]])))
    (is (equals (outer-product (array [[1] [2]]) (array [[3] [4]]) (array 2)) (array [[[[6] [8]]] [[[12] [16]]]]))))
  (testing "matrix * matrix * vector"
    (is (equals (outer-product (array [[1 2]]) (array [[3 4]]) (array [5 6]))
                (array [[[[[15 18] [20 24]]] [[[30 36] [40 48]]]]])))
    (is (equals (outer-product (array [[1 2]]) (array [5 6]) (array [[3 4]]))
                (array [[[[[15 20]] [[18 24]]] [[[30 40]] [[36 48]]]]])))
    (is (equals (outer-product (array [5 6]) (array [[1 2]]) (array [[3 4]]))
                (array [[[[[15 20]] [[30 40]]]] [[[[18 24]] [[36 48]]]]])))
    (is (equals (outer-product (array [[1] [2]]) (array [[3] [4]]) (array [5 6]))
                (array [[[[[15 18]] [[20 24]]]] [[[[30 36]] [[40 48]]]]]))))
  (testing "matrix * matrix * vector * scalar"
    (is (equals (outer-product (array [[1 2]]) (array [[3 4]]) (array [5 6]) (array 2))
                (array [[[[[30 36] [40 48]]] [[[60 72] [80 96]]]]])))
    (is (equals (outer-product (array [[1 2]]) (array [[3 4]]) (array 2) (array [5 6]))
                (array [[[[[30 36] [40 48]]] [[[60 72] [80 96]]]]])))
    (is (equals (outer-product (array [[1 2]]) (array 2) (array [[3 4]]) (array [5 6]))
                (array [[[[[30 36] [40 48]]] [[[60 72] [80 96]]]]])))
    (is (equals (outer-product (array 2) (array [[1 2]]) (array [[3 4]]) (array [5 6]))
                (array [[[[[30 36] [40 48]]] [[[60 72] [80 96]]]]])))
    (is (equals (outer-product (array [[1 2]]) (array [5 6]) (array [[3 4]]) (array 2))
                (array [[[[[30 40]] [[36 48]]] [[[60 80]] [[72 96]]]]])))
    (is (equals (outer-product (array [5 6]) (array [[1 2]]) (array [[3 4]]) (array 2))
                (array [[[[[30 40]] [[60 80]]]] [[[[36 48]] [[72 96]]]]])))
    (is (equals (outer-product (array [[1] [2]]) (array [[3] [4]]) (array [5 6]) (array 2))
                (array [[[[[30 36]] [[40 48]]]] [[[[60 72]] [[80 96]]]]])))
    (is (equals (outer-product (array [[1 2]]) (array [5 6]) (array 2) (array [[3 4]]))
                (array [[[[[30 40]] [[36 48]]] [[[60 80]] [[72 96]]]]])))
    (is (equals (outer-product (array [5 6]) (array 2) (array [[1 2]]) (array [[3 4]]))
                (array [[[[[30 40]] [[60 80]]]] [[[[36 48]] [[72 96]]]]])))
    (is (equals (outer-product (array 2) (array [[1] [2]]) (array [[3] [4]]) (array [5 6]))
                (array [[[[[30 36]] [[40 48]]]] [[[[60 72]] [[80 96]]]]]))))
  (testing "matrix * matrix * matrix"
    (is (equals (outer-product (array [[1 2]]) (array [[3 4]]) (array [[5 6]]))
                (array [[[[[[15 18]] [[20 24]]]] [[[[30 36]] [[40 48]]]]]])))
    (is (equals (outer-product (array [[1 2]]) (array [[5] [6]]) (array [[3 4]]))
                (array [[[[[[15 20]]] [[[18 24]]]] [[[[30 40]]] [[[36 48]]]]]]))))
  (testing "matrix * matrix * matrix * scalar"
    (is (equals (outer-product (array [[1 2]]) (array [[3 4]]) (array [[5 6]]) (array 2))
                (array [[[[[[30 36]] [[40 48]]]] [[[[60 72]] [[80 96]]]]]])))
    (is (equals (outer-product (array [[1 2]]) (array [[5] [6]]) (array [[3 4]]) (array 2))
                (array [[[[[[30 40]]] [[[36 48]]]] [[[[60 80]]] [[[72 96]]]]]])))))

(deftest eq!-test
  (is (thrown? Exception (eq! (array 2) (array 3))))
  (is (thrown? Exception (eq! (array 2))))
  (let [a (array [1 2 3])
        result (eq! a)]
    (is (equals a (array [1 1 1])))
    (is (equals result a)))
  (is (equals (eq! (array [[1 2] [3 4]])) (array [[1 1] [1 1]])))
  (let [a (array [1 2 3])
        result (eq! a (array 2))]
    (is (equals a (array [0 1 0])))
    (is (equals result a)))
  (let [a (array [1 2 3])
        b (array [0 2 2.5])
        result (eq! a b)]
    (is (equals result (array [0 1 0])))
    (is (equals a result))
    (is (equals b (array [0 2 2.5])))
    (is (= (rank result) 1)))
  (let [a (array [[6 2 3] [5 2 4]])
        b (array [[1 2 3]])]
    (eq! a b)
    (is (equals a (array [[0 1 1] [0 1 0]])))
    (is (equals b (array [[1 2 3]]))))
  (is (thrown? Exception (eq! (array [[1 2 3]]) (array [[6 2 3] [5 2 4]]))))
  (is (equals (eq! (array [[6 2 3] [5 2 1]]) (array [[2] [1]])) (array [[0 1 0] [0 0 1]])))
  (is (equals (eq! (array [[6 2 3] [5 2 1]]) (array [2 1])) (array [[0 1 0] [0 0 1]])))
  (is (equals (eq! (array [[6 2 3] [5 2 4]]) (array [1 2 3])) (array [[0 1 1] [0 1 0]])))
  (is (equals (eq! (array [[6 2 3] [5 2 4]]) (array 2)) (array [[0 1 0] [0 1 0]])))
  (is (equals (eq! (array [[6 2 3] [5 2 4]]) (array [[6 0 4] [-5 1 4]])) (array [[1 0 0] [0 0 1]])))
  (is (equals (eq! (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]) (array [[6 0 4] [-5 1 4]]))
              (array [[[1 0 0] [0 0 1]] [[1 0 1] [0 0 1]]])))
  (is (equals (eq! (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]) (array [[6 0 4]]))
              (array [[[1 0 0] [0 0 1]] [[1 0 1] [0 1 1]]])))
  (is (equals (eq! (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]) (array [6 0 4]))
              (array [[[1 0 0] [0 0 1]] [[1 0 1] [0 1 1]]])))
  (is (equals (eq! (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]) (array [[6 0]]))
              (array [[[1 0 0] [0 0 0]] [[1 0 0] [1 1 0]]])))
  (is (equals (eq! (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]) (array [[6] [0]]))
              (array [[[1 0 0] [0 0 0]] [[1 0 0] [1 1 0]]])))
  (is (equals (eq! (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]) (array [6 0]))
              (array [[[1 0 0] [0 0 0]] [[1 0 0] [1 1 0]]])))
  (is (equals (eq! (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]) (array [[6 0] [2 4]]))
              (array [[[1 0 0] [0 0 0]] [[0 1 0] [0 0 1]]])))
  (is (equals (eq! (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]])
                   (array [[[6 0 3] [0 0 4]] [[2 4 4] [1 0 4]]]))
              (array [[[1 0 1] [0 0 1]] [[0 0 1] [0 1 1]]]))))

(deftest eq-test
  (is (equals (eq (array 3)) (array 1)))
  (is (equals (eq (array [[1 2 3] [4 5 6]])) (array [[1 1 1] [1 1 1]])))
  (is (equals (eq (array [[[1 2 3] [4 5 6]] [[1 2 3] [4 5 6]]]))
              (array [[[1 1 1] [1 1 1]] [[1 1 1] [1 1 1]]])))
  (is (equals (eq (array 2) (array 3)) (array 0)))
  (is (equals (eq (array 2) (array 2)) (array 1)))
  (let [a (array [1 2 3])]
    (is (equals (eq a) (array [1 1 1])))
    (is (equals a (array [1 2 3]))))
  (let [a (array [1 2 3])]
    (is (equals (eq a (array 2)) (array [0 1 0])))
    (is (equals a (array [1 2 3]))))
  (is (equals (eq (array 2) (array [1 2 3])) (array [0 1 0])))
  (let [a (array [1 2 3])
        b (array [0 1 3])]
    (is (equals (eq a b) (array [0 0 1])))
    (is (equals a (array [1 2 3])))
    (is (equals b (array [0 1 3]))))
  (let [a (array [[6 2 3] [5 2 4]])
        b (array [[1 2 3]])]
    (is (equals (eq a b) (array [[0 1 1] [0 1 0]])))
    (is (equals a (array [[6 2 3] [5 2 4]])))
    (is (equals b (array [[1 2 3]]))))
  (is (equals (eq (array [[1 2 3]]) (array [[6 2 3] [5 2 4]])) (array [[0 1 1] [0 1 0]])))
  (is (equals (eq (array [[6 2 3] [5 2 1]]) (array [[2] [1]])) (array [[0 1 0] [0 0 1]])))
  (is (equals (eq (array [[2] [1]]) (array [[6 2 3] [5 2 1]])) (array [[0 1 0] [0 0 1]])))
  (is (equals (eq (array [[6 2 3] [5 2 1]]) (array [2 1])) (array [[0 1 0] [0 0 1]])))
  (is (equals (eq (array [2 1]) (array [[6 2 3] [5 2 1]])) (array [[0 1 0] [0 0 1]])))
  (is (equals (eq (array [[6 2 3] [5 2 4]]) (array [1 2 3])) (array [[0 1 1] [0 1 0]])))
  (is (equals (eq (array [1 2 3]) (array [[6 2 3] [5 2 4]])) (array [[0 1 1] [0 1 0]])))
  (is (equals (eq (array [[6 2 3] [5 2 4]]) (array 2)) (array [[0 1 0] [0 1 0]])))
  (is (equals (eq (array 2) (array [[6 2 3] [5 2 4]])) (array [[0 1 0] [0 1 0]])))
  (is (equals (eq (array [[6 2 3] [5 2 4]]) (array [[6 0 4] [-5 1 4]])) (array [[1 0 0] [0 0 1]])))
  (is (equals (eq (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]) (array [[6 0 4] [-5 1 4]]))
              (array [[[1 0 0] [0 0 1]] [[1 0 1] [0 0 1]]])))
  (is (equals (eq (array [[6 0 4] [-5 1 4]]) (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]))
              (array [[[1 0 0] [0 0 1]] [[1 0 1] [0 0 1]]])))
  (is (equals (eq (array [[6 0 4]]) (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]))
              (array [[[1 0 0] [0 0 1]] [[1 0 1] [0 1 1]]])))
  (is (equals (eq (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]) (array [6 0 4]))
              (array [[[1 0 0] [0 0 1]] [[1 0 1] [0 1 1]]])))
  (is (equals (eq (array [6 0 4]) (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]))
              (array [[[1 0 0] [0 0 1]] [[1 0 1] [0 1 1]]])))
  (is (equals (eq (array [[6 0]]) (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]))
              (array [[[1 0 0] [0 0 0]] [[1 0 0] [1 1 0]]])))
  (is (equals (eq (array [[6] [0]]) (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]))
              (array [[[1 0 0] [0 0 0]] [[1 0 0] [1 1 0]]])))
  (is (equals (eq (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]) (array [6 0]))
              (array [[[1 0 0] [0 0 0]] [[1 0 0] [1 1 0]]])))
  (is (equals (eq (array [6 0]) (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]))
              (array [[[1 0 0] [0 0 0]] [[1 0 0] [1 1 0]]])))
  (is (equals (eq (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]) (array [[6 0] [2 4]]))
              (array [[[1 0 0] [0 0 0]] [[0 1 0] [0 0 1]]])))
  (is (equals (eq (array [[6 0] [2 4]]) (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]))
              (array [[[1 0 0] [0 0 0]] [[0 1 0] [0 0 1]]])))
  (is (equals (eq (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]])
                  (array [[[6 0 3] [0 0 4]] [[2 4 4] [1 0 4]]]))
              (array [[[1 0 1] [0 0 1]] [[0 0 1] [0 1 1]]]))))

(deftest ne!-test
  (is (thrown? Exception (ne! (array 2) (array 3))))
  (is (thrown? Exception (ne! (array 2))))
  (let [a (array [1 2 3])
        result (ne! a)]
    (is (equals a (array [0 0 0])))
    (is (equals result a)))
  (is (equals (ne! (array [[1 2] [3 4]])) (array [[0 0] [0 0]])))
  (let [a (array [1 2 3])
        result (ne! a (array 2))]
    (is (equals a (array [1 0 1])))
    (is (equals result a)))
  (let [a (array [1 2 3])
        b (array [0 2 2.5])
        result (ne! a b)]
    (is (equals result (array [1 0 1])))
    (is (equals a result))
    (is (equals b (array [0 2 2.5])))
    (is (= (rank result) 1)))
  (let [a (array [[1 2 0] [5 1 3]])
        b (array [[1 2 3]])]
    (ne! a b)
    (is (equals a (array [[0 0 1] [1 1 0]])))
    (is (equals b (array [[1 2 3]]))))
  (is (equals (ne! (array [[6 2 3] [5 2 4]]) (array [1 2 3])) (array [[1 0 0] [1 0 1]])))
  (is (equals (ne! (array [[6 2 3] [5 2 4]]) (array [[6 0 4] [-5 1 4]])) (array [[0 1 1] [1 1 0]])))
  (is (equals (ne! (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]) (array [[6 0 4] [-5 1 4]]))
              (array [[[0 1 1] [1 1 0]] [[0 1 0] [1 1 0]]])))
  (is (equals (ne! (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]])
                   (array [[[6 0 3] [0 0 4]] [[2 4 4] [1 0 4]]]))
              (array [[[0 1 0] [1 1 0]] [[1 1 0] [1 0 0]]]))))

(deftest ne-test
  (is (equals (ne (array 3)) (array 0)))
  (is (equals (ne (array 3) (array 3)) (array 0)))
  (is (equals (ne (array 3) (array 7)) (array 1)))
  (let [a (array [1 2 3])]
    (is (equals (ne a) (array [0 0 0])))
    (is (equals a (array [1 2 3]))))
  (is (equals (ne (array [[1 2] [3 4]])) (array [[0 0] [0 0]])))
  (let [a (array [1 2 3])]
    (is (equals (ne a (array 2)) (array [1 0 1])))
    (is (equals a (array [1 2 3]))))
  (let [a (array [1 2 3])
        b (array [0 2 2.5])
        result (ne a b)]
    (is (equals result (array [1 0 1])))
    (is (equals a (array [1 2 3])))
    (is (equals b (array [0 2 2.5])))
    (is (= (rank result) 1)))
  (let [a (array [[1 2 0] [5 1 3]])
        b (array [[1 2 3]])]
    (is (equals (ne a b) (array [[0 0 1] [1 1 0]])))
    (is (equals a (array [[1 2 0] [5 1 3]])))
    (is (equals b (array [[1 2 3]]))))
  (is (equals (ne (array [[6 2 3] [5 2 4]]) (array [1 2 3])) (array [[1 0 0] [1 0 1]])))
  (is (equals (ne (array [1 2 3]) (array [[6 2 3] [5 2 4]])) (array [[1 0 0] [1 0 1]])))
  (is (equals (ne (array [[6 2 3] [5 2 4]]) (array [[6 0 4] [-5 1 4]])) (array [[0 1 1] [1 1 0]])))
  (is (equals (ne (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]) (array [[6 0 4] [-5 1 4]]))
              (array [[[0 1 1] [1 1 0]] [[0 1 0] [1 1 0]]])))
  (is (equals (ne (array [[6 0 4] [-5 1 4]]) (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]))
              (array [[[0 1 1] [1 1 0]] [[0 1 0] [1 1 0]]])))
  (is (equals (ne (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]])
                  (array [[[6 0 3] [0 0 4]] [[2 4 4] [1 0 4]]]))
              (array [[[0 1 0] [1 1 0]] [[1 1 0] [1 0 0]]]))))

(deftest gt!-test
  (is (thrown? Exception (gt! (array 2) (array 3))))
  (is (thrown? Exception (gt! (array 2))))
  (let [a (array [1 2 3])
        result (gt! a)]
    (is (equals a (array [1 1 1])))
    (is (equals result a)))
  (is (equals (gt! (array [[1 2] [3 4]])) (array [[1 1] [1 1]])))
  (let [a (array [1 2 3])
        result (gt! a (array 2))]
    (is (equals a (array [0 0 1])))
    (is (equals result a)))
  (let [a (array [1 2 3])
        b (array [0 2 4])
        result (gt! a b)]
    (is (equals result (array [1 0 0])))
    (is (equals a result))
    (is (equals b (array [0 2 4])))
    (is (= (rank result) 1)))
  (let [a (array [[1 2 0] [5 1 3]])
        b (array [[1 2 3]])]
    (gt! a b)
    (is (equals a (array [[0 0 0] [1 0 0]])))
    (is (equals b (array [[1 2 3]]))))
  (is (equals (gt! (array [[6 2 3] [5 2 4]]) (array [1 2 3])) (array [[1 0 0] [1 0 1]])))
  (is (equals (gt! (array [[6 2 3] [5 2 4]]) (array [[6 0 4] [-5 1 4]])) (array [[0 1 0] [1 1 0]])))
  (is (equals (gt! (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]) (array [[6 0 4] [-5 1 4]]))
              (array [[[0 1 0] [1 1 0]] [[0 1 0] [1 0 0]]])))
  (is (equals (gt! (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]])
                   (array [[[6 0 3] [0 0 4]] [[2 4 4] [1 0 4]]]))
              (array [[[0 1 0] [1 1 0]] [[1 0 0] [0 0 0]]]))))

(deftest gt-test
  (is (equals (gt (array 3)) (array 1)))
  (let [a (array [1 2 3])]
    (is (equals (gt a) (array [1 1 1])))
    (is (equals a (array [1 2 3]))))
  (is (equals (gt (array [[1 2] [3 4]])) (array [[1 1] [1 1]])))
  (is (equals (gt (array [[[1 2]] [[3 4]]])) (array [[[1 1]] [[1 1]]])))
  (is (equals (gt (array 3) (array 7)) (array 0)))
  (is (equals (gt (array 7) (array 3)) (array 1)))
  (is (equals (gt (array 3) (array 3)) (array 0)))
  (let [a (array [1 2 3])]
    (is (equals (gt a (array 2)) (array [0 0 1])))
    (is (equals a (array [1 2 3]))))
  (is (equals (gt (array 2) (array [1 2 3])) (array [1 0 0])))
  (let [a (array [1 2 3])
        b (array [0 2 4])]
    (is (equals (gt a b) (array [1 0 0])))
    (is (equals a (array [1 2 3])))
    (is (equals b (array [0 2 4]))))
  (let [a (array [[1 2 0] [5 1 3]])
        b (array [[1 2 3]])]
    (is (equals (gt a b) (array [[0 0 0] [1 0 0]])))
    (is (equals a (array [[1 2 0] [5 1 3]])))
    (is (equals b (array [[1 2 3]]))))
  (is (equals (gt (array [[1 2 3]]) (array [[1 2 0] [5 1 3]])) (array [[0 0 1] [0 1 0]])))
  (is (equals (gt (array [[6 2 3] [5 2 4]]) (array [1 2 3])) (array [[1 0 0] [1 0 1]])))
  (is (equals (gt (array [1 2 3]) (array [[6 1 3] [5 2 2]])) (array [[0 1 0] [0 0 1]])))
  (is (equals (gt (array [[6 2 3] [5 2 4]]) (array [[6 0 4] [-5 1 4]])) (array [[0 1 0] [1 1 0]])))
  (is (equals (gt (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]) (array [[6 0 4] [-5 1 4]]))
              (array [[[0 1 0] [1 1 0]] [[0 1 0] [1 0 0]]])))
  (is (equals (gt (array [[6 0 4] [-5 1 4]]) (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]))
              (array [[[0 0 1] [0 0 0]] [[0 0 0] [0 1 0]]])))
  (is (equals (gt (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]])
                  (array [[[6 0 3] [0 0 4]] [[2 4 4] [1 0 4]]]))
              (array [[[0 1 0] [1 1 0]] [[1 0 0] [0 0 0]]]))))

(deftest lt!-test
  (is (thrown? Exception (lt! (array 2) (array 3))))
  (is (thrown? Exception (lt! (array 2))))
  (let [a (array [1 2 3])
        result (lt! a)]
    (is (equals a (array [1 1 1])))
    (is (equals result a)))
  (is (equals (lt! (array [[1 2] [3 4]])) (array [[1 1] [1 1]])))
  (let [a (array [1 2 3])
        result (lt! a (array 2))]
    (is (equals a (array [1 0 0])))
    (is (equals result a)))
  (let [a (array [1 2 3])
        b (array [0 2 4])
        result (lt! a b)]
    (is (equals result (array [0 0 1])))
    (is (equals a result))
    (is (equals b (array [0 2 4])))
    (is (= (rank result) 1)))
  (let [a (array [[1 2 0] [5 1 3]])
        b (array [[1 2 3]])]
    (lt! a b)
    (is (equals a (array [[0 0 1] [0 1 0]])))
    (is (equals b (array [[1 2 3]]))))
  (is (equals (lt! (array [[-2 2 3] [5 2 0]]) (array [1 2 3])) (array [[1 0 0] [0 0 1]])))
  (is (equals (lt! (array [[6 0 4] [-5 1 4]]) (array [[6 2 3] [5 2 4]])) (array [[0 1 0] [1 1 0]])))
  (is (equals (lt! (array [[[5 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]) (array [[6 0 4] [-5 1 4]]))
              (array [[[1 0 1] [0 0 0]] [[0 0 0] [0 1 0]]])))
  (is (equals (lt! (array [[[6 0 3] [0 0 4]] [[2 4 4] [1 0 4]]])
                   (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]))
              (array [[[0 1 0] [1 1 0]] [[1 0 0] [0 0 0]]]))))

(deftest lt-test
  (is (equals (lt (array 3)) (array 1)))
  (let [a (array [1 2 3])]
    (is (equals (lt a) (array [1 1 1])))
    (is (equals a (array [1 2 3]))))
  (is (equals (lt (array [[1 2] [3 4]])) (array [[1 1] [1 1]])))
  (is (equals (lt (array [[[1 2]] [[3 4]]])) (array [[[1 1]] [[1 1]]])))
  (is (equals (lt (array 3) (array 7)) (array 1)))
  (is (equals (lt (array 7) (array 3)) (array 0)))
  (is (equals (lt (array 3) (array 3)) (array 0)))
  (let [a (array [1 2 3])]
    (is (equals (lt a (array 2)) (array [1 0 0])))
    (is (equals a (array [1 2 3]))))
  (is (equals (lt (array 2) (array [1 2 3])) (array [0 0 1])))
  (let [a (array [1 2 3])
        b (array [0 2 4])]
    (is (equals (lt a b) (array [0 0 1])))
    (is (equals a (array [1 2 3])))
    (is (equals b (array [0 2 4]))))
  (let [a (array [[1 2 0] [5 1 3]])
        b (array [[1 2 3]])]
    (is (equals (lt a b) (array [[0 0 1] [0 1 0]])))
    (is (equals a (array [[1 2 0] [5 1 3]])))
    (is (equals b (array [[1 2 3]]))))
  (is (equals (lt (array [[1 2 3]]) (array [[1 2 0] [5 1 3]])) (array [[0 0 0] [1 0 0]])))
  (is (equals (lt (array [1 2 3]) (array [[6 2 3] [5 2 4]])) (array [[1 0 0] [1 0 1]])))
  (is (equals (lt (array [[6 1 3] [5 2 2]]) (array [1 2 3])) (array [[0 1 0] [0 0 1]])))
  (is (equals (lt (array [[6 0 4] [-5 1 4]]) (array [[6 2 3] [5 2 4]])) (array [[0 1 0] [1 1 0]])))
  (is (equals (lt (array [[6 0 4] [-5 1 4]]) (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]))
              (array [[[0 1 0] [1 1 0]] [[0 1 0] [1 0 0]]])))
  (is (equals (lt (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]) (array [[6 0 4] [-5 1 4]]))
              (array [[[0 0 1] [0 0 0]] [[0 0 0] [0 1 0]]])))
  (is (equals (lt (array [[[6 0 3] [0 0 4]] [[2 4 4] [1 0 4]]])
                  (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]))
              (array [[[0 1 0] [1 1 0]] [[1 0 0] [0 0 0]]]))))

(deftest le!-test
  (is (thrown? Exception (le! (array 2) (array 3))))
  (is (thrown? Exception (le! (array 2))))
  (let [a (array [1 2 3])
        result (le! a)]
    (is (equals a (array [1 1 1])))
    (is (equals result a)))
  (is (equals (le! (array [[1 2] [3 4]])) (array [[1 1] [1 1]])))
  (let [a (array [1 2 3])
        result (le! a (array 2))]
    (is (equals a (array [1 1 0])))
    (is (equals result a)))
  (let [a (array [1 2 3])
        b (array [0 2 4])
        result (le! a b)]
    (is (equals result (array [0 1 1])))
    (is (equals a result))
    (is (equals b (array [0 2 4])))
    (is (= (rank result) 1)))
  (let [a (array [[1 2 0] [5 1 3]])
        b (array [[1 2 3]])]
    (le! a b)
    (is (equals a (array [[1 1 1] [0 1 1]])))
    (is (equals b (array [[1 2 3]]))))
  (is (equals (le! (array [[-2 2 3] [5 2 0]]) (array [1 2 3])) (array [[1 1 1] [0 1 1]])))
  (is (equals (le! (array [[6 0 4] [-5 1 4]]) (array [[6 2 3] [5 2 4]])) (array [[1 1 0] [1 1 1]])))
  (is (equals (le! (array [[[5 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]) (array [[6 0 4] [-5 1 4]]))
              (array [[[1 0 1] [0 0 1]] [[1 0 1] [0 1 1]]])))
  (is (equals (le! (array [[[6 0 3] [0 0 4]] [[2 4 4] [1 0 4]]])
                   (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]))
              (array [[[1 1 1] [1 1 1]] [[1 0 1] [0 1 1]]]))))

(deftest le-test
  (is (equals (le (array 3)) (array 1)))
  (let [a (array [1 2 3])]
    (is (equals (le a) (array [1 1 1])))
    (is (equals a (array [1 2 3]))))
  (is (equals (le (array [[1 2] [3 4]])) (array [[1 1] [1 1]])))
  (is (equals (le (array [[[1 2]] [[3 4]]])) (array [[[1 1]] [[1 1]]])))
  (is (equals (le (array 3) (array 7)) (array 1)))
  (is (equals (le (array 7) (array 3)) (array 0)))
  (is (equals (le (array 3) (array 3)) (array 1)))
  (let [a (array [1 2 3])]
    (is (equals (le a (array 2)) (array [1 1 0])))
    (is (equals a (array [1 2 3]))))
  (is (equals (le (array 2) (array [1 2 3])) (array [0 1 1])))
  (let [a (array [1 2 3])
        b (array [0 2 4])]
    (is (equals (le a b) (array [0 1 1])))
    (is (equals a (array [1 2 3])))
    (is (equals b (array [0 2 4]))))
  (let [a (array [[1 2 3]])
        b (array [[1 2 0] [5 1 3]])]
    (is (equals (le a b) (array [[1 1 0] [1 0 1]])))
    (is (equals b (array [[1 2 0] [5 1 3]])))
    (is (equals a (array [[1 2 3]]))))
  (is (equals (le (array [[1 2 0] [5 1 3]]) (array [[1 2 3]])) (array [[1 1 1] [0 1 1]])))
  (is (equals (le (array [[6 2 3] [0 2 4]]) (array [1 2 3])) (array [[0 1 1] [1 1 0]])))
  (is (equals (le (array [1 2 3]) (array [[6 1 3] [5 2 2]])) (array [[1 0 1] [1 1 0]])))
  (is (equals (le (array [[6 2 3] [5 2 4]]) (array [[6 0 4] [-5 1 4]])) (array [[1 0 1] [0 0 1]])))
  (is (equals (le (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]) (array [[6 0 4] [-5 1 4]]))
              (array [[[1 0 1] [0 0 1]] [[1 0 1] [0 1 1]]])))
  (is (equals (le (array [[6 0 4] [-5 1 4]]) (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]))
              (array [[[1 1 0] [1 1 1]] [[1 1 1] [1 0 1]]])))
  (is (equals (le (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]])
                  (array [[[6 0 3] [0 0 4]] [[2 4 4] [1 0 4]]]))
              (array [[[1 0 1] [0 0 1]] [[0 1 1] [1 1 1]]]))))

(deftest ge!-test
  (is (thrown? Exception (ge! (array 2) (array 3))))
  (is (thrown? Exception (ge! (array 2))))
  (let [a (array [1 2 3])
        result (ge! a)]
    (is (equals a (array [1 1 1])))
    (is (equals result a)))
  (is (equals (ge! (array [[1 2] [3 4]])) (array [[1 1] [1 1]])))
  (let [a (array [1 2 3])
        result (ge! a (array 2))]
    (is (equals a (array [0 1 1])))
    (is (equals result a)))
  (let [a (array [1 2 3])
        b (array [0 2 4])
        result (ge! a b)]
    (is (equals result (array [1 1 0])))
    (is (equals a result))
    (is (equals b (array [0 2 4])))
    (is (= (rank result) 1)))
  (let [a (array [[1 2 0] [5 1 3]])
        b (array [[1 2 3]])]
    (ge! a b)
    (is (equals a (array [[1 1 0] [1 0 1]])))
    (is (equals b (array [[1 2 3]]))))
  (is (equals (ge! (array [[-2 2 3] [5 2 0]]) (array [1 2 3])) (array [[0 1 1] [1 1 0]])))
  (is (equals (ge! (array [[6 0 4] [-5 1 4]]) (array [[6 2 3] [5 2 4]])) (array [[1 0 1] [0 0 1]])))
  (is (equals (ge! (array [[[5 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]) (array [[6 0 4] [-5 1 4]]))
              (array [[[0 1 0] [1 1 1]] [[1 1 1] [1 0 1]]])))
  (is (equals (ge! (array [[[6 0 3] [0 0 4]] [[2 4 4] [1 0 4]]])
                   (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]))
              (array [[[1 0 1] [0 0 1]] [[0 1 1] [1 1 1]]]))))

(deftest ge-test
  (is (equals (ge (array 3)) (array 1)))
  (let [a (array [1 2 3])]
    (is (equals (ge a) (array [1 1 1])))
    (is (equals a (array [1 2 3]))))
  (is (equals (ge (array [[1 2] [3 4]])) (array [[1 1] [1 1]])))
  (is (equals (ge (array [[[1 2]] [[3 4]]])) (array [[[1 1]] [[1 1]]])))
  (is (equals (ge (array 3) (array 7)) (array 0)))
  (is (equals (ge (array 7) (array 3)) (array 1)))
  (is (equals (ge (array 3) (array 3)) (array 1)))
  (let [a (array [1 2 3])]
    (is (equals (ge a (array 2)) (array [0 1 1])))
    (is (equals a (array [1 2 3]))))
  (is (equals (ge (array 2) (array [1 2 3])) (array [1 1 0])))
  (let [a (array [1 2 3])
        b (array [0 2 4])]
    (is (equals (ge a b) (array [1 1 0])))
    (is (equals a (array [1 2 3])))
    (is (equals b (array [0 2 4]))))
  (let [a (array [[1 2 3]])
        b (array [[1 2 0] [5 1 3]])]
    (is (equals (ge a b) (array [[1 1 1] [0 1 1]])))
    (is (equals b (array [[1 2 0] [5 1 3]])))
    (is (equals a (array [[1 2 3]]))))
  (is (equals (ge (array [[1 2 0] [5 1 3]]) (array [[1 2 3]])) (array [[1 1 0] [1 0 1]])))
  (is (equals (ge (array [[6 2 3] [0 2 4]]) (array [1 2 3])) (array [[1 1 1] [0 1 1]])))
  (is (equals (ge (array [1 2 3]) (array [[6 1 3] [5 2 2]])) (array [[0 1 1] [0 1 1]])))
  (is (equals (ge (array [[6 2 3] [5 2 4]]) (array [[6 0 4] [-5 1 4]])) (array [[1 1 0] [1 1 1]])))
  (is (equals (ge (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]])
                  (array [[6 0 4] [-5 1 4]]))
              (array [[[1 1 0] [1 1 1]] [[1 1 1] [1 0 1]]])))
  (is (equals (ge (array [[6 0 4] [-5 1 4]])
                  (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]))
              (array [[[1 0 1] [0 0 1]] [[1 0 1] [0 1 1]]])))
  (is (equals (ge (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]])
                  (array [[[6 0 3] [0 0 4]] [[2 4 4] [1 0 4]]]))
              (array [[[1 1 1] [1 1 1]] [[1 0 1] [0 1 1]]]))))

(deftest emax-test
  (is (equals (emax (array 3)) (array 3)))
  (is (equals (emax (array [-5 3 2])) (array 3)))
  (is (equals (emax (array [-5 -2 -3])) (array -2)))
  (is (equals (emax (array [[-5 3 2] [4 0 -20]])) (array 4)))
  (is (equals (emax (array [[[-5 3 2] [4 0 -20]] [[100 0 2] [3 4 5]]])) (array 100))))

(deftest emin-test
  (is (equals (emin (array 3)) (array 3)))
  (is (equals (emin (array [1 3 2])) (array 1)))
  (is (equals (emin (array [[-5 3 2] [4 0 -20]])) (array -20)))
  (is (equals (emin (array [[[-5 3 2] [4 0 3]] [[100 0 2] [3 4 5]]])) (array -5))))

(deftest esum-test
  (is (equals (esum (array 7)) (array 7)))
  (is (equals (esum (array [1 5 3])) (array 9)))
  (is (equals (esum (array [[1 5 3] [-3 -2 20]])) (array 24)))
  (is (equals (esum (array [[[1 5 3] [-3 -2 20]] [[1 1 1] [0 0 0]]])) (array 27))))

(deftest emean-test
  (is (equals (emean (array 7)) (array 7)))
  (is (equals (emean (array [1 5 3])) (array 3)))
  (is (equals (emean (array [[1 5 3] [-3 -2 20]])) (array 4)))
  (is (equals (emean (array [[[1 5 3] [-3 -2 20]] [[1 1 1] [0 0 0]]])) (array 2.25))))

(deftest estdev-test
  (is (equals (estdev (array 7)) (array 0)))
  (is (equals (estdev (array [1 5 3])) (array 2)))
  (is (equals (estdev (array [[1 5 3] [-3 -2 20]])) (array 8.3904707)))
  (is (equals (estdev (array [[[1 5 3] [-3 -2 20]] [[1 1 1] [0 0 0]]])) (array 5.9562801))))

(deftest max-test
  (is (equals (max (array 3)) (array 3)))
  (let [a (array [1 2 3])]
    (is (equals (max a) (array [1 2 3])))
    (is (equals a (array [1 2 3]))))
  (is (equals (max (array [[1 2] [3 4]])) (array [[1 2] [3 4]])))
  (is (equals (max (array [[[1 2]] [[3 4]]])) (array [[[1 2]] [[3 4]]])))
  (is (equals (max (array 3) (array 7)) (array 7)))
  (is (equals (max (array 7) (array 3)) (array 7)))
  (is (equals (max (array 3) (array 3)) (array 3)))
  (is (equals (max (array 3) (array 7) (array 2)) (array 7)))
  (let [a (array [1 2 3])]
    (is (equals (max a (array 2)) (array [2 2 3])))
    (is (equals a (array [1 2 3]))))
  (is (equals (max (array 2) (array [1 2 3])) (array [2 2 3])))
  (let [a (array [1 2 3])
        b (array [0 2 4])]
    (is (equals (max a b) (array [1 2 4])))
    (is (equals a (array [1 2 3])))
    (is (equals b (array [0 2 4]))))
  (is (equals (max (array [4 2 3 7]) (array [5 1 3 7]) (array [4 0 6 7]))
              (array [5 2 6 7])))
  (let [a (array [[1 2 3]])
        b (array [[1 2 0] [5 1 3]])]
    (is (equals (max a b) (array [[1 2 3] [5 2 3]])))
    (is (equals b (array [[1 2 0] [5 1 3]])))
    (is (equals a (array [[1 2 3]]))))
  (is (equals (max (array [[1 2 0] [5 1 3]]) (array [[1 2 3]])) (array [[1 2 3] [5 2 3]])))
  (is (equals (max (array [[6 2 3] [0 2 4]]) (array [1 2 3])) (array [[6 2 3] [1 2 4]])))
  (is (equals (max (array [1 2 3]) (array [[6 1 3] [5 2 2]])) (array [[6 2 3] [5 2 3]])))
  (is (equals (max (array [[6 2 3] [5 2 4]]) (array [[6 0 4] [-5 1 4]])) (array [[6 2 4] [5 2 4]])))
  (is (equals (max (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]])
                   (array [[6 0 4] [-5 1 4]]))
              (array [[[6 2 4] [5 2 4]] [[6 2 4] [0 1 4]]])))
  (is (equals (max (array [[6 0 4] [-5 1 4]])
                   (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]))
              (array [[[6 2 4] [5 2 4]] [[6 2 4] [0 1 4]]])))
  (is (equals (max (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]])
                   (array [[[6 0 3] [0 0 4]] [[2 4 4] [1 0 4]]]))
              (array [[[6 2 3] [5 2 4]] [[6 4 4] [1 0 4]]]))))

(deftest min-test
  (is (equals (min (array 3)) (array 3)))
  (let [a (array [1 2 3])]
    (is (equals (min a) (array [1 2 3])))
    (is (equals a (array [1 2 3]))))
  (is (equals (min (array [[1 2] [3 4]])) (array [[1 2] [3 4]])))
  (is (equals (min (array [[[1 2]] [[3 4]]])) (array [[[1 2]] [[3 4]]])))
  (is (equals (min (array 3) (array 7)) (array 3)))
  (is (equals (min (array 7) (array 3)) (array 3)))
  (is (equals (min (array 3) (array 3)) (array 3)))
  (is (equals (min (array 3) (array 7) (array 2)) (array 2)))
  (let [a (array [1 2 3])]
    (is (equals (min a (array 2)) (array [1 2 2])))
    (is (equals a (array [1 2 3]))))
  (is (equals (min (array 2) (array [1 2 3])) (array [1 2 2])))
  (let [a (array [1 2 3])
        b (array [0 2 4])]
    (is (equals (min a b) (array [0 2 3])))
    (is (equals a (array [1 2 3])))
    (is (equals b (array [0 2 4]))))
  (is (equals (min (array [0 2 3 7]) (array [5 1 3 7]) (array [4 2 2 7]))
              (array [0 1 2 7])))
  (let [a (array [[1 2 3]])
        b (array [[1 2 0] [5 1 3]])]
    (is (equals (min a b) (array [[1 2 0] [1 1 3]])))
    (is (equals b (array [[1 2 0] [5 1 3]])))
    (is (equals a (array [[1 2 3]]))))
  (is (equals (min (array [[1 2 0] [5 1 3]]) (array [[1 2 3]])) (array [[1 2 0] [1 1 3]])))
  (is (equals (min (array [[6 2 3] [0 2 4]]) (array [1 2 3])) (array [[1 2 3] [0 2 3]])))
  (is (equals (min (array [1 2 3]) (array [[6 1 3] [5 2 2]])) (array [[1 1 3] [1 2 2]])))
  (is (equals (min (array [[6 2 3] [5 2 4]]) (array [[6 0 4] [-5 1 4]]))
              (array [[6 0 3] [-5 1 4]])))
  (is (equals (min (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]])
                   (array [[6 0 4] [-5 1 4]]))
              (array [[[6 0 3] [-5 1 4]] [[6 0 4] [-5 0 4]]])))
  (is (equals (min (array [[6 0 4] [-5 1 4]])
                   (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]]))
              (array [[[6 0 3] [-5 1 4]] [[6 0 4] [-5 0 4]]])))
  (is (equals (min (array [[[6 2 3] [5 2 4]] [[6 2 4] [0 0 4]]])
                   (array [[[6 0 3] [0 0 4]] [[2 4 4] [1 0 4]]]))
              (array [[[6 0 3] [0 0 4]] [[2 2 4] [0 0 4]]]))))

(deftest abs!-test
  (testing "modifies the argument in-place"
    (let [nd (array [[-1 2 -3] [4 -5 6]])
          result (abs! nd)]
      (is (equals result (array [[1 2 3] [4 5 6]])))
      (is (equals nd result))))
  (testing "returns the correct result for arguments of all dimensionalities"
    (is (equals (abs! (array -3)) (array 3)))
    (is (equals (abs! (array 3)) (array 3)))
    (is (equals (abs! (array [-1 2 -3])) (array [1 2 3])))
    (is (equals (abs! (array [[-1 2 -3] [4 -5 6]])) (array [[1 2 3] [4 5 6]])))
    (is (equals (abs! (array [[[-1 2] [-3 4]] [[-5 6] [-7 8]]])) (array [[[1 2] [3 4]] [[5 6] [7 8]]])))))

(deftest abs-test
  (testing "does not mutate the argument"
    (let [nd (array [[-1 2 -3] [4 -5 6]])
          result (abs nd)]
      (is (equals result (array [[1 2 3] [4 5 6]])))
      (is (equals nd (array [[-1 2 -3] [4 -5 6]])))))
  (testing "returns the correct result for arguments of all dimensionalities"
    (is (equals (abs (array -3)) (array 3)))
    (is (equals (abs (array 3)) (array 3)))
    (is (equals (abs (array [-1 2 -3])) (array [1 2 3])))
    (is (equals (abs (array [[-1 2 -3] [4 -5 6]])) (array [[1 2 3] [4 5 6]])))
    (is (equals (abs (array [[[-1 2] [-3 4]] [[-5 6] [-7 8]]])) (array [[[1 2] [3 4]] [[5 6] [7 8]]])))))

(deftest sum-along-test
  (testing "scalar"
    (is (equals (sum-along (array 7) [] false) (array 7)))
    (is (equals (sum-along (array 7) []) (array 7)))
    (is (equals (sum-along (array 7) [] true) (array 7)))
    (is (equals (sum-along (array 7) nil false) (array 7)))
    (is (equals (sum-along (array 7) nil) (array 7)))
    (is (equals (sum-along (array 7) nil true) (array 7)))
    (is (thrown? Exception (sum-along (array 7) [0] false))))
  (testing "vector"
    (is (equals (sum-along (array [1 2 3]) [0] false) (array [6])))
    (is (equals (sum-along (array [1 2 3]) 0 false) (array [6])))
    (is (equals (sum-along (array [1 2 3]) [0] true) (array 6)))
    (is (equals (sum-along (array [1 2 3]) [0]) (array 6)))
    (is (equals (sum-along (array [1 2 3]) 0 true) (array 6)))
    (is (equals (sum-along (array [1 2 3]) 0) (array 6)))
    (is (equals (sum-along (array [1 2 3]) [] true) (array [1 2 3])))
    (is (equals (sum-along (array [1 2 3]) []) (array [1 2 3])))
    (is (equals (sum-along (array [1 2 3]) [] false) (array [1 2 3])))
    (is (equals (sum-along (array [1 2 3]) nil true) (array [1 2 3])))
    (is (equals (sum-along (array [1 2 3]) nil) (array [1 2 3])))
    (is (equals (sum-along (array [1 2 3]) nil false) (array [1 2 3])))
    (is (thrown? Exception (sum-along (array [1 2 3]) [1])))
    (is (thrown? Exception (sum-along (array [1 2 3]) [0 1]))))
  (testing "matrix"
    (is (equals (sum-along (array [[1 2 3] [4 5 6]]) [0] false) (array [[5 7 9]])))
    (is (equals (sum-along (array [[1 2 3] [4 5 6]]) [0] true) (array [5 7 9])))
    (is (equals (sum-along (array [[1 2 3] [4 5 6]]) [0]) (array [5 7 9])))
    (is (equals (sum-along (array [[1 2 3] [4 5 6]]) [1] false) (array [[6] [15]])))
    (is (equals (sum-along (array [[1 2 3] [4 5 6]]) [1] true) (array [6 15])))
    (is (equals (sum-along (array [[1 2 3] [4 5 6]]) [1]) (array [6 15])))
    (is (equals (sum-along (array [[1 2 3] [4 5 6]]) [0 1] false) (array [[21]])))
    (is (equals (sum-along (array [[1 2 3] [4 5 6]]) [0 1] true) (array 21)))
    (is (equals (sum-along (array [[1 2 3] [4 5 6]]) [0 1]) (array 21)))
    (is (equals (sum-along (array [[1 2 3] [4 5 6]]) [] false) (array [[1 2 3] [4 5 6]])))
    (is (equals (sum-along (array [[1 2 3] [4 5 6]]) [] true) (array [[1 2 3] [4 5 6]])))
    (is (equals (sum-along (array [[1 2 3] [4 5 6]]) []) (array [[1 2 3] [4 5 6]])))
    (is (thrown? Exception (sum-along (array [1 2 3]) [2]))))
  (testing "tensors"
    (let [t (array [[[1 2 3] [4 5 6]]
                    [[7 8 9] [10 11 12]]])]
      (is (equals (sum-along t [0] false) (array [[[8 10 12] [14 16 18]]])))
      (is (equals (sum-along t [0] true) (array [[8 10 12] [14 16 18]])))
      (is (equals (sum-along t [1] false) (array [[[5 7 9]] [[17 19 21]]])))
      (is (equals (sum-along t [1] true) (array [[5 7 9] [17 19 21]])))
      (is (equals (sum-along t [2] false) (array [[[6] [15]] [[24] [33]]])))
      (is (equals (sum-along t [2] true) (array [[6 15] [24 33]])))
      (is (equals (sum-along t [0 1] false) (array [[[22 26 30]]])))
      (is (equals (sum-along t [0 1] true) (array [22 26 30])))
      (is (equals (sum-along t [0 2] false) (array [[[30] [48]]])))
      (is (equals (sum-along t [0 2] true) (array [30 48])))
      (is (equals (sum-along t [1 2] false) (array [[[21]] [[57]]])))
      (is (equals (sum-along t [1 2] true) (array [21 57])))
      (is (equals (sum-along t [0 1 2] false) (array [[[78]]])))
      (is (equals (sum-along t [0 1 2] true) (array 78))))))

(deftest min-along-test
  (testing "scalar"
    (is (equals (min-along (array 7) [] false) (array 7)))
    (is (equals (min-along (array 7) []) (array 7)))
    (is (equals (min-along (array 7) [] true) (array 7)))
    (is (equals (min-along (array 7) nil false) (array 7)))
    (is (equals (min-along (array 7) nil) (array 7)))
    (is (equals (min-along (array 7) nil true) (array 7)))
    (is (thrown? Exception (min-along (array 7) [0] false))))
  (testing "vector"
    (is (equals (min-along (array [3 1 1 2]) [0] false) (array [1])))
    (is (equals (min-along (array [4 2 3 2]) 0 false) (array [2])))
    (is (equals (min-along (array [-2 3 0]) [0] true) (array -2)))
    (is (equals (min-along (array [2 3 0]) [0]) (array 0)))
    (is (equals (min-along (array [7]) 0 true) (array 7)))
    (is (equals (min-along (array [4 2 3 2]) 0) (array 2)))
    (is (equals (min-along (array [1 2 3]) [] true) (array [1 2 3])))
    (is (equals (min-along (array [1 2 3]) []) (array [1 2 3])))
    (is (equals (min-along (array [1 2 3]) [] false) (array [1 2 3])))
    (is (equals (min-along (array [1 2 3]) nil true) (array [1 2 3])))
    (is (equals (min-along (array [1 2 3]) nil) (array [1 2 3])))
    (is (equals (min-along (array [1 2 3]) nil false) (array [1 2 3])))
    (is (thrown? Exception (min-along (array [1 2 3]) [1])))
    (is (thrown? Exception (min-along (array [1 2 3]) [0 1]))))
  (testing "matrix"
    (is (equals (min-along (array [[3 2 2] [4 2 1]]) [0] false) (array [[3 2 1]])))
    (is (equals (min-along (array [[3 2 2] [4 2 1]]) [0] true) (array [3 2 1])))
    (is (equals (min-along (array [[3 2 2] [4 2 1]]) [0]) (array [3 2 1])))
    (is (equals (min-along (array [[2 4 2] [0 3 5]]) [1] false) (array [[2] [0]])))
    (is (equals (min-along (array [[2 4 2] [0 3 5]]) [1] true) (array [2 0])))
    (is (equals (min-along (array [[2 4 2] [0 3 5]]) [1]) (array [2 0])))
    (is (equals (min-along (array [[2 3 1] [5 2 4]]) [0 1] false) (array [[1]])))
    (is (equals (min-along (array [[2 3 1] [5 2 4]]) [0 1] true) (array 1)))
    (is (equals (min-along (array [[1 3 1] [5 2 0]]) [0 1]) (array 0)))
    (is (equals (min-along (array [[1 3 1] [5 2 0]]) [] false) (array [[1 3 1] [5 2 0]])))
    (is (equals (min-along (array [[1 3 1] [5 2 0]]) [] true) (array [[1 3 1] [5 2 0]])))
    (is (equals (min-along (array [[1 3 1] [5 2 0]]) []) (array [[1 3 1] [5 2 0]])))
    (is (equals (min-along (array [[5 2 0]]) [0]) (array [5 2 0])))
    (is (equals (min-along (array [[5 2 0]]) [0] false) (array [[5 2 0]])))
    (is (equals (min-along (array [[5 2 0]]) [1]) (array [0])))
    (is (equals (min-along (array [[5 2 0]]) [1] false) (array [[0]])))
    (is (equals (min-along (array [[5 2 0]]) [0 1]) (array 0)))
    (is (equals (min-along (array [[5 2 0]]) [0 1] false) (array [[0]])))
    (is (equals (min-along (array [[5] [2] [0]]) [0]) (array [0])))
    (is (equals (min-along (array [[5] [2] [0]]) [0] false) (array [[0]])))
    (is (equals (min-along (array [[5] [2] [0]]) [1]) (array [5 2 0])))
    (is (equals (min-along (array [[5] [2] [0]]) [1] false) (array [[5] [2] [0]])))
    (is (equals (min-along (array [[5] [2] [0]]) [0 1]) (array 0)))
    (is (equals (min-along (array [[5] [2] [0]]) [0 1] false) (array [[0]])))
    (is (thrown? Exception (min-along (array [1 2 3]) [2]))))
  (testing "tensors"
    (let [t (array [[[4 2 3] [1 2 4]]
                    [[3 2 4] [5 0 4]]])]
      (is (equals (min-along t [0] false) (array [[[3 2 3] [1 0 4]]])))
      (is (equals (min-along t [0] true) (array [[3 2 3] [1 0 4]])))
      (is (equals (min-along t [1] false) (array [[[1 2 3]] [[3 0 4]]])))
      (is (equals (min-along t [1] true) (array [[1 2 3] [3 0 4]])))
      (is (equals (min-along t [2] false) (array [[[2] [1]] [[2] [0]]])))
      (is (equals (min-along t [2] true) (array [[2 1] [2 0]])))
      (is (equals (min-along t [0 1] false) (array [[[1 0 3]]])))
      (is (equals (min-along t [0 1] true) (array [1 0 3])))
      (is (equals (min-along t [0 2] false) (array [[[2] [0]]])))
      (is (equals (min-along t [0 2] true) (array [2 0])))
      (is (equals (min-along t [1 2] false) (array [[[1]] [[0]]])))
      (is (equals (min-along t [1 2] true) (array [1 0])))
      (is (equals (min-along t [0 1 2] false) (array [[[0]]])))
      (is (equals (min-along t [0 1 2] true) (array 0))))))

(deftest max-along-test
  (testing "scalar"
    (is (equals (max-along (array 7) [] false) (array 7)))
    (is (equals (max-along (array 7) []) (array 7)))
    (is (equals (max-along (array 7) [] true) (array 7)))
    (is (equals (max-along (array 7) nil false) (array 7)))
    (is (equals (max-along (array 7) nil) (array 7)))
    (is (equals (max-along (array 7) nil true) (array 7)))
    (is (thrown? Exception (max-along (array 7) [0] false))))
  (testing "vector"
    (is (equals (max-along (array [3 4 4 2]) [0] false) (array [4])))
    (is (equals (max-along (array [4 5 3 5]) 0 false) (array [5])))
    (is (equals (max-along (array [4 3 0]) [0] true) (array 4)))
    (is (equals (max-along (array [2 3 5]) [0]) (array 5)))
    (is (equals (max-along (array [7]) 0 true) (array 7)))
    (is (equals (max-along (array [4 2 3 4]) 0) (array 4)))
    (is (equals (max-along (array [1 2 3]) [] true) (array [1 2 3])))
    (is (equals (max-along (array [1 2 3]) []) (array [1 2 3])))
    (is (equals (max-along (array [1 2 3]) [] false) (array [1 2 3])))
    (is (equals (max-along (array [1 2 3]) nil true) (array [1 2 3])))
    (is (equals (max-along (array [1 2 3]) nil) (array [1 2 3])))
    (is (equals (max-along (array [1 2 3]) nil false) (array [1 2 3])))
    (is (thrown? Exception (max-along (array [1 2 3]) [1])))
    (is (thrown? Exception (max-along (array [1 2 3]) [0 1]))))
  (testing "matrix"
    (is (equals (max-along (array [[2 3 3] [1 3 4]]) [0] false) (array [[2 3 4]])))
    (is (equals (max-along (array [[2 3 3] [1 3 4]]) [0] true) (array [2 3 4])))
    (is (equals (max-along (array [[2 3 3] [1 3 4]]) [0]) (array [2 3 4])))
    (is (equals (max-along (array [[3 1 3] [5 2 0]]) [1] false) (array [[3] [5]])))
    (is (equals (max-along (array [[3 1 3] [5 2 0]]) [1] true) (array [3 5])))
    (is (equals (max-along (array [[3 1 3] [5 2 0]]) [1]) (array [3 5])))
    (is (equals (max-along (array [[3 2 4] [0 3 1]]) [0 1] false) (array [[4]])))
    (is (equals (max-along (array [[3 2 4] [0 3 1]]) [0 1] true) (array 4)))
    (is (equals (max-along (array [[4 2 4] [0 3 5]]) [0 1]) (array 5)))
    (is (equals (max-along (array [[4 2 4] [0 3 5]]) [] false) (array [[4 2 4] [0 3 5]])))
    (is (equals (max-along (array [[4 2 4] [0 3 5]]) [] true) (array [[4 2 4] [0 3 5]])))
    (is (equals (max-along (array [[4 2 4] [0 3 5]]) []) (array [[4 2 4] [0 3 5]])))
    (is (equals (max-along (array [[5 2 0]]) [0]) (array [5 2 0])))
    (is (equals (max-along (array [[5 2 0]]) [0] false) (array [[5 2 0]])))
    (is (equals (max-along (array [[5 2 0]]) [1]) (array [5])))
    (is (equals (max-along (array [[5 2 0]]) [1] false) (array [[5]])))
    (is (equals (max-along (array [[5 2 0]]) [0 1]) (array 5)))
    (is (equals (max-along (array [[5 2 0]]) [0 1] false) (array [[5]])))
    (is (equals (max-along (array [[5] [2] [0]]) [0]) (array [5])))
    (is (equals (max-along (array [[5] [2] [0]]) [0] false) (array [[5]])))
    (is (equals (max-along (array [[5] [2] [0]]) [1]) (array [5 2 0])))
    (is (equals (max-along (array [[5] [2] [0]]) [1] false) (array [[5] [2] [0]])))
    (is (equals (max-along (array [[5] [2] [0]]) [0 1]) (array 5)))
    (is (equals (max-along (array [[5] [2] [0]]) [0 1] false) (array [[5]])))
    (is (thrown? Exception (max-along (array [1 2 3]) [2]))))
  (testing "tensors"
    (let [t (array [[[1 3 2] [4 3 1]]
                    [[2 3 1] [0 5 1]]])]
      (is (equals (max-along t [0] false) (array [[[2 3 2] [4 5 1]]])))
      (is (equals (max-along t [0] true) (array [[2 3 2] [4 5 1]])))
      (is (equals (max-along t [1] false) (array [[[4 3 2]] [[2 5 1]]])))
      (is (equals (max-along t [1] true) (array [[4 3 2] [2 5 1]])))
      (is (equals (max-along t [2] false) (array [[[3] [4]] [[3] [5]]])))
      (is (equals (max-along t [2] true) (array [[3 4] [3 5]])))
      (is (equals (max-along t [0 1] false) (array [[[4 5 2]]])))
      (is (equals (max-along t [0 1] true) (array [4 5 2])))
      (is (equals (max-along t [0 2] false) (array [[[3] [5]]])))
      (is (equals (max-along t [0 2] true) (array [3 5])))
      (is (equals (max-along t [1 2] false) (array [[[4]] [[5]]])))
      (is (equals (max-along t [1 2] true) (array [4 5])))
      (is (equals (max-along t [0 1 2] false) (array [[[5]]])))
      (is (equals (max-along t [0 1 2] true) (array 5))))))

(deftest argmax-along-test
  (testing "scalar"
    (is (equals (argmax-along (array 3) nil) (array 0)))
    (is (equals (argmax-along (array 3) []) (array 0))))
  (testing "vector"
    (is (equals (argmax-along (array [2 3 4 1]) 0) (array 2)))
    (is (equals (argmax-along (array [2 4 4 1]) 0) (array 1)))
    (is (equals (argmax-along (array [2 4 3 1]) nil) (array 1)))
    (is (equals (argmax-along (array [4 2 3 1]) nil) (array 0))))
  (testing "matrix"
    (is (equals (argmax-along (array [[1 3 2] [4 3 1]]) 0) (array [1 0 0])))
    (is (equals (argmax-along (array [[1 3 2] [4 3 1]]) [0]) (array [1 0 0])))
    (is (equals (argmax-along (array [[1 3 2] [4 3 1]]) [0 0]) (array [1 0 0])))
    (is (equals (argmax-along (array [[1 3 2] [4 3 1]]) [1]) (array [1 0])))
    (is (equals (argmax-along (array [[4 2 4] [0 3 5]]) [1]) (array [0 2])))
    (is (equals (argmax-along (array [[3 2 4] [0 3 1]]) [0 1]) (array 2)))
    (is (equals (argmax-along (array [[3 2 4] [0 3 1]]) [1 0 1 0]) (array 2)))
    (is (equals (argmax-along (array [[4 2 4] [0 3 5]]) [0 1]) (array 5)))
    (is (equals (argmax-along (array [[4 2 4] [0 3 5]]) []) (array 5)))
    (is (equals (argmax-along (array [[4 2 4] [0 3 5]]) nil) (array 5)))
    (is (equals (argmax-along (array [[0 3 5]]) [1]) (array [2])))
    (is (equals (argmax-along (array [[0 3 5]]) 1) (array [2])))
    (is (equals (argmax-along (array [[0 3 5]]) [0]) (array [0 0 0])))
    (is (equals (argmax-along (array [[0 3 5]]) 0) (array [0 0 0])))
    (is (equals (argmax-along (array [[0] [5] [3]]) [0]) (array [1])))
    (is (equals (argmax-along (array [[0] [5] [3]]) [1]) (array [0 0 0]))))
  (testing "tensor"
    (is (equals (argmax-along (array [[[1 3 2] [4 3 1]]
                                      [[2 3 1] [0 5 1]]]) 0)
                (array [[1 0 0] [0 1 0]])))
    (is (equals (argmax-along (array [[[1 3 2] [4 3 1]]
                                      [[2 3 1] [0 5 1]]]) 1)
                (array [[1 0 0] [0 1 0]])))
    (is (equals (argmax-along (array [[[1 3 2] [4 3 1]]
                                      [[2 3 1] [0 5 1]]]) [1])
                (array [[1 0 0] [0 1 0]])))
    (is (equals (argmax-along (array [[[1 3 2] [4 3 1]]
                                      [[2 3 1] [0 5 1]]]) 2)
                (array [[1 0] [1 1]])))
    (is (equals (argmax-along (array [[[1 3 2] [4 3 1]]
                                      [[2 3 1] [0 5 1]]]) [0 1])
                (array [1 3 0])))
    (is (equals (argmax-along (array [[[1 3 2] [4 3 1]]
                                      [[2 3 1] [0 5 1]]]) [0 2])
                (array [1 4])))
    (is (equals (argmax-along (array [[[1 3 2] [4 3 1]]
                                      [[2 3 1] [0 5 1]]]) [1 2])
                (array [3 4])))
    (is (equals (argmax-along (array [[[1 3 2] [4 3 1]]
                                      [[2 3 1] [0 5 1]]]) [0 1 2])
                (array 10)))
    (is (equals (argmax-along (array [[[1 3 2] [4 3 1]]
                                      [[2 3 1] [0 5 1]]]) [])
                (array 10)))
    (is (equals (argmax-along (array [[[1 3 2] [4 3 1]]
                                      [[2 3 1] [0 5 1]]]) nil)
                (array 10))))
  (testing "throws exceptions with invalid axes"
    (is (thrown? Exception (argmax-along (array [[1 3 2] [4 3 1]]) 2)))
    (is (thrown? Exception (argmax-along (array [[1 3 2] [4 3 1]]) [2])))
    (is (thrown? Exception (argmax-along (array [[1 3 2] [4 3 1]]) 1.2)))
    (is (thrown? Exception (argmax-along (array [[1 3 2] [4 3 1]]) [-1])))))

(deftest argmin-along-test
  (testing "scalar"
    (is (equals (argmin-along (array 3) nil) (array 0)))
    (is (equals (argmin-along (array 3) []) (array 0))))
  (testing "vector"
    (is (equals (argmin-along (array [2 3 4 1]) 0) (array 3)))
    (is (equals (argmin-along (array [2 1 1 3]) 0) (array 1)))
    (is (equals (argmin-along (array [0 2 3 1]) nil) (array 0)))
    (is (equals (argmin-along (array [4 2 3 0]) nil) (array 3))))
  (testing "matrix"
    (is (equals (argmin-along (array [[3 2 2] [4 2 1]]) 0) (array [0 0 1])))
    (is (equals (argmin-along (array [[3 2 2] [4 2 1]]) [0]) (array [0 0 1])))
    (is (equals (argmin-along (array [[3 2 2] [4 2 1]]) [0 0]) (array [0 0 1])))
    (is (equals (argmin-along (array [[4 3 2] [4 1 1]]) [1]) (array [2 1])))
    (is (equals (argmin-along (array [[2 4 2] [0 3 5]]) [1]) (array [0 0])))
    (is (equals (argmin-along (array [[2 3 1] [5 2 4]]) [0 1]) (array 2)))
    (is (equals (argmin-along (array [[2 3 1] [5 2 4]]) [1 0 1 0]) (array 2)))
    (is (equals (argmin-along (array [[1 3 1] [5 2 0]]) [0 1]) (array 5)))
    (is (equals (argmin-along (array [[1 3 1] [5 2 0]]) []) (array 5)))
    (is (equals (argmin-along (array [[1 3 1] [5 2 0]]) nil) (array 5)))
    (is (equals (argmin-along (array [[5 2 0]]) [1]) (array [2])))
    (is (equals (argmin-along (array [[5 2 0]]) 1) (array [2])))
    (is (equals (argmin-along (array [[5 2 0]]) [0]) (array [0 0 0])))
    (is (equals (argmin-along (array [[5 2 0]]) 0) (array [0 0 0])))
    (is (equals (argmin-along (array [[5] [2] [0]]) [0]) (array [2])))
    (is (equals (argmin-along (array [[5] [2] [0]]) [1]) (array [0 0 0]))))
  (testing "tensor"
    (is (equals (argmin-along (array [[[4 2 3] [1 2 4]]
                                      [[3 2 4] [5 0 4]]]) 0)
                (array [[1 0 0] [0 1 0]])))
    (is (equals (argmin-along (array [[[4 2 3] [1 2 4]]
                                      [[3 2 4] [5 0 4]]]) 1)
                (array [[1 0 0] [0 1 0]])))
    (is (equals (argmin-along (array [[[4 2 3] [1 2 4]]
                                      [[3 2 4] [5 0 4]]]) [1])
                (array [[1 0 0] [0 1 0]])))
    (is (equals (argmin-along (array [[[4 2 3] [1 2 4]]
                                      [[3 2 4] [5 0 4]]]) 2)
                (array [[1 0] [1 1]])))
    (is (equals (argmin-along (array [[[4 2 3] [1 2 4]]
                                      [[3 2 4] [5 0 4]]]) [0 1])
                (array [1 3 0])))
    (is (equals (argmin-along (array [[[4 2 3] [1 2 4]]
                                      [[3 2 4] [5 0 4]]]) [0 2])
                (array [1 4])))
    (is (equals (argmin-along (array [[[4 2 3] [1 2 4]]
                                      [[3 2 4] [5 0 4]]]) [1 2])
                (array [3 4])))
    (is (equals (argmin-along (array [[[4 2 3] [1 2 4]]
                                      [[3 2 4] [5 0 4]]]) [0 1 2])
                (array 10)))
    (is (equals (argmin-along (array [[[4 2 3] [1 2 4]]
                                      [[3 2 4] [5 0 4]]]) [])
                (array 10)))
    (is (equals (argmin-along (array [[[4 2 3] [1 2 4]]
                                      [[3 2 4] [5 0 4]]]) nil)
                (array 10))))
  (testing "throws exceptions with invalid axes"
    (is (thrown? Exception (argmin-along (array [[1 3 2] [4 3 1]]) 2)))
    (is (thrown? Exception (argmin-along (array [[1 3 2] [4 3 1]]) [2])))
    (is (thrown? Exception (argmin-along (array [[1 3 2] [4 3 1]]) 1.2)))
    (is (thrown? Exception (argmin-along (array [[1 3 2] [4 3 1]]) [-1])))))

(deftest permute-test
  (testing "scalar"
    (let [n (array 7)
          result (permute n [])]
      (is (equals result (array 7)))
      (is (empirically-same-data? n result)))
    (is (equals (permute (array 7) nil) (array 7))))
  (testing "vector"
    (let [n (array [1 2 3])
          result (permute n [0])]
      (is (equals result (array [1 2 3])))
      (is (empirically-same-data? n result))))
  (testing "matrix"
    (let [n (array [[1 2 3] [4 5 6]])
          result (permute n [0 1])]
      (is (equals result (array [[1 2 3] [4 5 6]])))
      (is (empirically-same-data? n result)))
    (let [n (array [[1 2 3] [4 5 6]])
          result (permute n [1 0])]
      (is (equals result (array [[1 4] [2 5] [3 6]])))
      (is (equals n (array [[1 2 3] [4 5 6]])))
      (is (empirically-same-data? n result))))
  (testing "tensor"
    (let [n (array [[[1 2 3] [4 5 6]]
                    [[7 8 9] [10 11 12]]])
          original-n (clone n)]
      (let [result (permute n [0 1 2])]
        (is (equals result (array [[[1 2 3] [4 5 6]]
                                   [[7 8 9] [10 11 12]]])))
        (is (empirically-same-data? n result)))
      (let [result (permute n [0 2 1])]
        (is (equals result (array [[[1 4] [2 5] [3 6]]
                                   [[7 10] [8 11] [9 12]]])))
        (is (equals n original-n))
        (is (empirically-same-data? n result)))
      (let [result (permute n [1 0 2])]
        (is (equals result (array [[[1 2 3] [7 8 9]]
                                   [[4 5 6] [10 11 12]]])))
        (is (equals n original-n))
        (is (empirically-same-data? n result)))
      (let [result (permute n [1 2 0])]
        (is (equals result (array [[[1 7] [2 8] [3 9]] [[4 10] [5 11] [6 12]]])))
        (is (equals n original-n))
        (is (empirically-same-data? n result)))
      (let [result (permute n [2 0 1])]
        (is (equals result (array [[[1 4] [7 10]] [[2 5] [8 11]] [[3 6] [9 12]]])))
        (is (equals n original-n))
        (is (empirically-same-data? n result)))
      (let [result (permute n [2 1 0])]
        (is (equals result (array [[[1 7] [4 10]] [[2 8] [5 11]] [[3 9] [6 12]]])))
        (is (equals n original-n))
        (is (empirically-same-data? n result))))))

(deftest permute!-test
  (testing "scalar"
    (let [n (array 7)
          result (permute! n [])]
      (is (equals result (array 7)))
      (is (= result n))
      (is (empirically-same-data? n result)))
    (is (equals (permute! (array 7) nil) (array 7))))
  (testing "vector"
    (let [n (array [1 2 3])
          result (permute! n [0])]
      (is (equals result (array [1 2 3])))
      (is (= result n))
      (is (empirically-same-data? n result))))
  (testing "matrix"
    (let [n (array [[1 2 3] [4 5 6]])
          result (permute! n [0 1])]
      (is (equals result (array [[1 2 3] [4 5 6]])))
      (is (empirically-same-data? n result))
      (is (= result n)))
    (let [n (array [[1 2 3] [4 5 6]])
          result (permute! n [1 0])]
      (is (equals result (array [[1 4] [2 5] [3 6]])))
      (is (equals n result))
      (is (= n result))
      (is (empirically-same-data? n result))))
  (testing "tensor"
    (let [n (array [[[1 2 3] [4 5 6]]
                    [[7 8 9] [10 11 12]]])
          result (permute! n [0 1 2])]
      (is (equals result (array [[[1 2 3] [4 5 6]]
                                 [[7 8 9] [10 11 12]]])))
      (is (equals n result))
      (is (= n result))
      (is (empirically-same-data? n result)))
    (let [n (array [[[1 2 3] [4 5 6]]
                    [[7 8 9] [10 11 12]]])
          result (permute! n [0 2 1])]
      (is (equals result (array [[[1 4] [2 5] [3 6]]
                                 [[7 10] [8 11] [9 12]]])))
      (is (equals n result))
      (is (= n result))
      (is (empirically-same-data? n result)))
    (let [n (array [[[1 2 3] [4 5 6]]
                    [[7 8 9] [10 11 12]]])
          result (permute! n [1 0 2])]
      (is (equals result (array [[[1 2 3] [7 8 9]]
                                 [[4 5 6] [10 11 12]]])))
      (is (equals n result))
      (is (= n result))
      (is (empirically-same-data? n result)))
    (let [n (array [[[1 2 3] [4 5 6]]
                    [[7 8 9] [10 11 12]]])
          result (permute! n [1 2 0])]
      (is (equals result (array [[[1 7] [2 8] [3 9]] [[4 10] [5 11] [6 12]]])))
      (is (equals n result))
      (is (= n result))
      (is (empirically-same-data? n result)))
    (let [n (array [[[1 2 3] [4 5 6]]
                    [[7 8 9] [10 11 12]]])
          result (permute! n [2 0 1])]
      (is (equals result (array [[[1 4] [7 10]] [[2 5] [8 11]] [[3 6] [9 12]]])))
      (is (equals n result))
      (is (= n result))
      (is (empirically-same-data? n result)))
    (let [n (array [[[1 2 3] [4 5 6]]
                    [[7 8 9] [10 11 12]]])
          result (permute! n [2 1 0])]
      (is (equals result (array [[[1 7] [4 10]] [[2 8] [5 11]] [[3 9] [6 12]]])))
      (is (equals n result))
      (is (= n result))
      (is (empirically-same-data? n result)))))

(deftest same-data?-test
  (let [a (array [1 2 3 4])
        b (array [1 2 3 4])]
    (testing "returns true for the same object"
      (is (same-data? a a)))
    (testing "returns true for views of the same data"
      (is (same-data? a (transpose a)))
      (is (same-data? (transpose a) (transpose a)))
      (is (same-data? a (reshape a [2 2])))
      (is (same-data? (reshape a [2 2]) (reshape a [2 2])))
      (is (same-data? a (select-range a :first)))
      (is (same-data? (select-range a :first) (select-range a :first)))
      (is (same-data? a (submatrix a 0 [0 2])))
      (is (same-data? (submatrix a 0 [0 2]) (submatrix a 0 [0 2]))))
    (testing "returns false for different objects"
      (is (not (same-data? a b))))
    (testing "returns false for cloned objects"
      (is (not (same-data? a (clone a)))))
    (testing "returns false for views of different objects"
      (is (not (same-data? a (transpose b))))
      (is (not (same-data? (transpose a) (transpose b)))))))

#_(clojure.test/run-tests)
