(ns tensure.test-utils
  (:require [tensure.core :as m]
            [tensure.utils :as u])
  (:import [tensure.core Tensure]))

(defn empirically-same-data?
  "Returns true iff `a` and `b` are tensors over the same data. Assumes that sharing of data is all-or-none."
  [^Tensure a ^Tensure b]
  (let [indices (repeat (m/rank a) 0)
        val (m/clone (apply m/select-range a indices))
        new-val (m/mul val (m/array 1.1))
        _ (apply m/set-range! a (concat indices [new-val]))
        b-updated? (= (apply m/select-range b indices)
                      new-val)]
    (apply m/set-range! a (concat indices [val]))
    b-updated?))

(defn array-seqs=
  "Returns true iff `a` and `b` are seqs of the same length with elements in matched position that are
  equal according to `m/equals`, using the given tolerance."
  ([a b]
   (array-seqs= a b nil))
  ([a b tolerance]
   (let [compare-fn (if tolerance
                      #(m/equals %1 %2 tolerance)
                      m/equals)]
     (and (count a)
          (count b)
          (every? (fn [[a-el b-el]]
                    (compare-fn a-el b-el))
                  (u/zip a b))))))
