(ns tensure.core
  "A high-level tensor math library for Clojure."
  (:require [clojure.pprint]
            [tensure.utils :as u])
  (:refer-clojure :exclude [min max])
  (:import [org.nd4j.linalg.api.buffer DataBuffer$Type FloatBuffer]
           [org.nd4j.linalg.api.iter NdIndexIterator]
           [org.nd4j.linalg.api.ndarray INDArray]
           [org.nd4j.linalg.factory Nd4j]
           [org.nd4j.linalg.indexing NDArrayIndex INDArrayIndex]
           [org.nd4j.linalg.ops.transforms Transforms]))

(declare stringify-neurond shape)

#_(set! *warn-on-reflection* true)
(def ^:dynamic *max-array-print-size* 500)

; TODO: Support double precision.
(Nd4j/setDataType DataBuffer$Type/FLOAT)

(defrecord Tensure
  [^org.nd4j.linalg.cpu.nativecpu.NDArray o rank]
  Object
  (toString [nd] (stringify-neurond nd)))

(defn- o
  "Given a Tensure obejct, returns the corresponding underlying implementation tensor object."
  ^org.nd4j.linalg.cpu.nativecpu.NDArray [^Tensure nd]
  (.o nd))

(defn- ->o
  "Returns `o`, an underlying tensor implementation object, with a type hint added."
  ^org.nd4j.linalg.cpu.nativecpu.NDArray [o]
  o)

(defn rank
  "Returns the `rank` of `nd`. Equivalent to `dimensionality`."
  [^Tensure nd]
  (.rank nd))

; TODO: Print vectors without double closing braces (single instead) and just generally improve how tensors
; are printed.
(defn stringify-neurond
  ^String [^Tensure nd]
  (str "#" "Tensure\n"
       (let [o (o nd)]
         (if (<= (.length o) *max-array-print-size*)
           (.toString o)
           (str "Large array of shape " (shape nd))))))
(. ^clojure.lang.MultiFn clojure.pprint/simple-dispatch addMethod Tensure (comp print stringify-neurond))

(defmethod print-method Tensure [nd ^java.io.Writer w]
  (.write w (stringify-neurond nd)))

; TODO: Write a better method for pretty-printing a tensor.
(def pm println)

(defn array?
  "Returns `true` iff `o` is an tensure array."
  [o]
  (instance? Tensure o))

(defn- seq-shape
  "Returns the shape of an nd-array composed of clojure sequences. Does not check for well-formedness."
  [s]
  (when (and (sequential? s) (seq s))
    (cons (count s) (seq-shape (first s)))))

(defn clone
  "Copies the data underlying `nd` and returns a new nd array referencing the copied data."
  [^Tensure nd]
  (Tensure. (.dup (o nd)) (rank nd)))

(defn array
  "Returns a new tensor containing `data`, a number, a tensor, or a Clojure data structure consisting of a
  seq of seqs or numbers. If a tensor is provided, a clone will be returned."
  [data]
  (cond (array? data) (clone data)
        (number? data) (Tensure. (Nd4j/scalar ^java.lang.Number data) 0)
        :else (let [d (float-array (flatten data))
                    shape (int-array (seq-shape data))]
                (when-not (= (count d) (apply * shape))
                  (u/throw-str "Cannot not construct a Tensure array from a clojure data structure "
                               "that does not represent a well-formed tensor."))
                (Tensure. (Nd4j/create d shape \c)
                          (count shape)))))

(defn same-data?
  "Returns true iff all arguments are tensors referencing the same underling data."
  [& nds]
  (->> nds
       (map #(-> % o .data))
       (reduce (fn [^FloatBuffer last ^FloatBuffer current]
                 (if (.sameUnderlyingData last current)
                   current
                   (reduced false))))
       boolean))

(defn scalar
  "Given number `n`, returns a new scalar."
  [n]
  (when-not (number? n)
    (u/throw-str "Cannot construct a scalar from a non-number."))
  (array n))

(def scalar-array
  "Equivalent to `scalar`, included only for `core.matrix` compatability."
  scalar)

(defn matrix
  "Given a two-dimensional collection of `data`, returns a new rank-2 tensor (matrix) containing those data.
  Equivalent to `array` but throws an `Exception` if `data` is not two-dimensional."
  [data]
  (when-not (and (sequential? data)
                 (sequential? (first data))
                 (number? (first (first data))))
    (u/throw-str ("Cannot construct matrix from non-2-dimensional data.")))
  (array data))

(defn shape
  "Returns a Clojure vector representing the `shape` of `nd`."
  [^Tensure nd]
  (case (int (rank nd))
    0 nil
    1 [(.columns (o nd))]
    (vec (.shape (o nd)))))

(defn scalar?
  "Returns `true` iff `nd` is a rank-0 tensure tensor (i.e. a scalar)."
  [^Tensure nd]
  (= (rank nd) 0))

(defn vec?
  "Returns `true` iff `nd` is a rank-1 tensure tensor (i.e. a vector)."
  [^Tensure nd]
  (= (rank nd) 1))

(defn matrix?
  "Returns `true` iff `nd` is a rank-2 tensure tensor (i.e. a matrix)."
  [^Tensure nd]
  (= (rank nd) 2))

(defn scalar->number
  "Returns a `java.lang.Float` that's numerically equivalent to scalar `nd`."
  ^java.lang.Float [^Tensure nd]
  (when-not (scalar? nd)
    (u/throw-str "Cannot convert nonscalar to a number."))
  (.getFloat (o nd) (int-array [0])))

(defn ->number
  "Returns a number that's numerically equivalent to `n`, which may be a Tensure tensor or a clojure/Java
  number of any type."
  [n]
  (if (array? n)
    (scalar->number n)
    n))

(defn ->int
  "Like `->number` but coerces the return value to an integer."
  [n]
  (int (->number n)))

(defn transpose
  "Returns the tensor transpose of `nd`."
  [nd]
  (if (<= (rank nd) 1)
    nd
    (Tensure.
      (.transpose (o nd))
      (rank nd))))

(defn reshape
  "Returns a tensor, possibly representing a view over the same data as `nd`, with shape `shape`.
  `(apply * shape)` must equal `(ecount nd)` (i.e. the number of elements in the original tensor and the
  output tensor must be the same). The order of elements in the input vector is retained in row major order
  in the output tensor."
  [nd shape]
  (let [rank (count shape)]
    (Tensure.
      (.reshape (o nd) (int-array (if (= rank 1)
                                    [1 (first shape)]
                                    shape)))
      rank)))

(defn permute
  "Returns a new tensor (possibly a view) in which the dimensions of `nd` have been reordered as specified in
  `axes`, a seq of axis indices. `axes` must contain every natural number in [0, rank of nd). The result and
  `nd` share data but will have different shapes."
  [^Tensure nd axes]
  (case (int (rank nd))
    0 (if-not (seq axes)
        nd
        (u/throw-str "Invalid permutation of a scalar: " (vec axes)))
    1 (if (= axes [0])
        nd
        (u/throw-str "Invalid permutation of a vector: " (vec axes)))
    (Tensure.
      (.permute (o nd) (int-array axes))
      (rank nd))))

(defn permute!
  "Like `permuted-view` but mutates `nd` rather than returning a new view."
  [^Tensure nd axes]
  (case (int (rank nd))
    0 (when (seq axes)
        (u/throw-str "Invalid permutation of a scalar: " (vec axes)))
    1 (when (not= axes [0])
        (u/throw-str "Invalid permutation of a vector: " (vec axes)))
    (.permutei (o nd) (int-array axes)))
  nd)

(defn ecount
  "Returns the total number of elements in `nd`."
  [nd]
  (.length (o nd)))

(defn eseq
  "Returns a seq of `java.lang.Float`s representing the elements in `nd` in row-major order."
  [nd]
  (-> nd
      o
      .ravel
      .toFloatVector
      seq))

; TODO: Investigate alternatives to emap and emap-indexed for performance. These are very expensive, because
; they require moving the tensors between in and out of the JVM/device.
; TODO: Can possibly parallelize these easily with pmap.
(defn emap
  "Element-wise map over all elemends in one or more tensors, which must all be the same shape. Returns a new
  tensor of that shape that contains the result of applying `f` to the elements in the equivalent positions
  in all tensor arguments."
  [f ^Tensure nd & more]
  (let [nd-shape (shape nd)]
    (when-not (every? #(= nd-shape (shape %)) more)
      (u/throw-str "Invalid shapes. All arguments provided to `emap` must have the same shape."))
    (->> (map eseq (cons nd more))
         (apply map f)
         array
         (#(reshape % nd-shape)))))

(defn emap-indexed
  "Element-wise map over all elemends in one or more tensors. Returns a new tensor that's the same shape as
  `nd` and that contains the result of applying `f` to an index vector and all the elements in that position
  in all tensor arguemnts."
  [f ^Tensure nd & more]
  (let [nd-shape (shape nd)
        iter (NdIndexIterator. \c (.shape (o nd)))]
    (when-not (every? #(= nd-shape (shape %)) more)
      (u/throw-str "Invalid shapes. All arguments provided to `emap-indexed` must have the same shape."))
    (->> (map eseq (cons nd more))
         (cons (repeatedly (ecount nd) #(vec (.next iter))))
         (apply map f)
         array
         (#(reshape % nd-shape)))))

(defn array->vector
  "Returns a clojure PersistentVector representation of `nd`, or a number if `nd` is a scalar."
  [nd]
  (if (scalar? nd)
    (scalar->number nd)
    (first (reduce
             #(mapv vec (partition %2 %1))
             (eseq nd)
             (reverse (shape nd))))))

(def dimensionality
  "Alias for `rank` provided for compatability with `core.matrix`."
  rank)

(defn dimension-count
  "Returns the size of the dimension corresponding to `dim-index`."
  ^long [nd dim-index]
  (let [dim-index (->> (if (array? dim-index)
                         (scalar->number dim-index)
                         dim-index)
                       int)]
    (case (int (rank nd))
      0 0
      1 (if (= 0 dim-index)
          (.size (o nd) 1)
          0)
      (.size (o nd) dim-index))))

(defn column-count
  "Returns the size of the second axis of `nd`. This is equal to the number of columns in `nd` if `nd` is a
  matrix. Equivalent to `(dimension-count nd 1)`."
  [^Tensure nd]
  (dimension-count nd 1))

(defn row-count
  "Returns the size of the first axis of `nd`. This is equal to the number of rows in `nd` if `nd` is a
  matrix. Equivalent to `(dimension-count nd 0)`."
  [^Tensure nd]
  (dimension-count nd 0))

(defn zeros
  "Returns a new tensor of `shape` filled with 0's."
  [shape]
  (if (seq shape)
    (Tensure.
      (Nd4j/zeros (int-array shape))
      (count shape))
    (array 0)))

(defn ones
  "Returns a new tensor of `shape` filled with 1's."
  [shape]
  (if (seq shape)
    (let [nd (Nd4j/ones (int-array shape))
          rank (count shape)]
      (Tensure.
        (if (= 1 rank)
          ; The reshape is to compensate for an apparent bug in Nd4j wherein Nd4j/ones produces an array with
          ; an internal shape of one dimension for vectors (whereas Nd4j otherwise doesn't seem to have a
          ; concept of one-dimensional arrays).
          (.reshape nd (int-array (cons 1 shape)))
          nd)
        rank))
    (array 1)))

(defn fill!
  "Mutates the data underlying `nd` by setting every element to number `n`. Returns `nd`."
  [^Tensure nd ^java.lang.Number n]
  (.assign (o nd) n)
  nd)

(defn filled
  "Returns a new tensor of `shape` in which every element is number `n`."
  [shape n]
  (let [^Tensure result (zeros shape)
        ^java.lang.Float num (if (array? n)
                               (scalar->number n)
                               n)]
    (.assign (o result) num)
    result))

(defn assign!
  "Mutates the data underlying tensor `target` by setting every element in `target` to the value of the
  corresponding element in tensor `src`."
  [^Tensure target ^Tensure src]
  (.assign (o target) (o src))
  target)

(defn- get-random-generator-shape-array
  ^ints [shape]
  (->> (if (number? shape)
         (if (zero? shape)
           (u/throw-str "Cannot generate a tensor of size 0.")
           [shape])
         shape)
       int-array))

(defn set-rng-seed!
  "Sets the random number generator (RNG) seed used by all statistical sampling tensor creators (e.g.
  `sample-uniform`, etc."
  [seed]
  (.setSeed (Nd4j/getRandom) (long seed)))

(defn sample-uniform
  "Returns an array of `shape` with elements sampled uniformly from [0, 1) using the given random number
  generator `seed`. If `shape` is an integer, returns a vector of that length."
  ([shape seed]
   (when seed (set-rng-seed! seed))
   (sample-uniform shape))
  ([shape]
   (let [shape-array (get-random-generator-shape-array shape)]
     (Tensure.
       (Nd4j/rand shape-array)
       (count shape-array)))))

; TODO: Fix the seeded version of this. It's extremely slow.
(defn sample-normal
  "Returns an array of `shape` with elements drawn from a normal distribution with mean 0 and standard
  deviation 1 using the given random number generator `seed`. If `shape` is an integer, returns a vector of
  that length."
  ([shape seed]
   (when seed (set-rng-seed! seed))
   (sample-normal shape))
  ([shape]
   (let [shape-array (get-random-generator-shape-array shape)]
     (Tensure.
       (Nd4j/randn shape-array)
       (count shape-array)))))

(defn sample-rand-int
  "Returns an array of `shape` with elements drawn from a uniform integer distribution over [0, n) using the
  given random number generator `seed`. If `shape` is an integer, returns a vector of that length. The
  result's elements will be integer-valued but not necessarily of type Integer."
  ([shape ^java.lang.Long n]
   (sample-rand-int shape n nil))
  ([shape ^java.lang.Long n seed]
   (let [r (sample-uniform shape seed)
         or (o r)]
     (.muli or n)
     (Transforms/floor or false)
     r)))

(defn mset!
  "Sets the value of a particular element in `nd`. `more` is a seq like [...indices, value]. For example,
  `(mset! scalar 7)` sets a mutable scalar to 7, `(mset! vector 1 7)` sets the element at index 1 in a
  vector to 7, `(mset! matrix 1 3 7)` sets the element in the first row, third column of a matrix to 7,
  and so on. Mutates `nd`."
  [^Tensure nd & more]
  (let [^ints indices (int-array (drop-last more))
        ^float value (last more)]
    (.putScalar (o nd) indices value)
    nd))

(defn mset
  "Returns a clone of `nd` with the value of a particular element set to a certain value. `more` is a seq
  like [...indices, value]. For example, `(mset! scalar 7)` sets a mutable scalar to 7, `(mset! vector 1 7)`
  sets the element at index 1 in a vector to 7, `(mset! matrix 1 3 7)` sets the element in the first row,
  third column of a matrix to 7, and so on. Mutates `nd`."
  [^Tensure nd & more]
  (apply mset! (clone nd) more))

(defn- get-range-selection-indices
  [^Tensure nd selections]
  (let [selection-rank (count selections)
        ^"[Lorg.nd4j.linalg.indexing.INDArrayIndex;" index-array (make-array INDArrayIndex selection-rank)
        rank-reductions (atom 0)]
    (when (not= selection-rank (rank nd))
      (u/throw-str "Cannot select from '" selection-rank "' dimensions of an array with '"
                   (rank nd) "' dimensions."))
    (doseq [[i s] (u/zip (range selection-rank) selections)
            :let [index (cond (number? s) (do (swap! rank-reductions inc)
                                              (NDArrayIndex/point s))
                              (vector? s) (let [[start stop step] s
                                                stop (if (= start stop) (inc stop) stop)
                                                step (if step
                                                       ; This adjustment of step is compensation for a quirk
                                                       ; of Nd4j, where it gives an exception if there is
                                                       ; an interval start value and if start + step > stop.
                                                       (clojure.core/min (- stop start) step)
                                                       1)]
                                            (NDArrayIndex/interval (int start) (int step) (int stop)))
                              #_(case (count s)
                                  2 (NDArrayIndex/interval (first s) (last s))
                                  3 (NDArrayIndex/interval (first s) (second s) (last s)))
                                (= s :first) (do (swap! rank-reductions inc)
                                                 (NDArrayIndex/point 0))
                                (= s :last) (do (swap! rank-reductions inc)
                                                (NDArrayIndex/point (dec (dimension-count nd i))))
                                (= s :all) (NDArrayIndex/all)
                                (= s :butlast) (NDArrayIndex/interval 0 (dec (dimension-count nd i)))
                                (= s :rest) (NDArrayIndex/interval 1 (dimension-count nd i)))]]
      (aset index-array i index))
    [index-array (- (rank nd) @rank-reductions)]))

(defn select-range
  "Returns a view of a (not necessarily continuous) selection of `nd`. `selections` is a seq of objects
  indicating what range of the corresponding dimension should be selected. These objects can be:
    - a number - indicating a slice of that dimension (the dimension will be eliminated)
    - a two-element vector like [start stop] - indicating a range of slices in [start, stop)
    - a three-element vector like [start stop step] - indicating a range of every `step`th slice in
      [start, stop)
    - `:first` - indicating the first slice (the dimension will be eliminated)
    - `:last` - indicating the last slice (the dimension will be eliminated)
    - `:all` - indicating that all slices through that dimension should be kept
    - `:butlast` - indicating that all but the last slice of that dimension should be kept
    - `:rest` - indicating that all but the first slice of that dimension should be kept
  The size of `selections` should match the dimensionality of `nd`."
  [^Tensure nd & selections]
  (let [[^"[Lorg.nd4j.linalg.indexing.INDArrayIndex;" selection-indices
         new-rank] (get-range-selection-indices nd selections)
        result-o (.get (o nd) selection-indices)]
    (Tensure.
      (if (= 1 new-rank)
        (.reshape result-o (int-array [1 (.length result-o)]))
        result-o)
      new-rank)))

(defn select-axis-range
  "Like `select-range` but applies a selection to only a single axis (selects everything along other axes).
  `selection` must be a valid `selection` argument to `select-range`, and `axis` must be a valid axis index."
  [^Tensure nd axis selection]
  (->> (assoc (vec (repeat (rank nd) :all)) axis selection)
       (apply select-range nd)))

(defn set-range!
  "Like `select-range`, but mutates the data underlying the specified selection of `nd` by setting every
  element to the corresponding value in the last argument, a source tensor. I.e., arguments should be:
    (set-range! target-tensor selection-for-axis-0 selection-for-axis-1 ... source-tensor)
  `source-tensor` must have the same shape as the selection from `target-tensor`."
  [^Tensure nd & args]
  (let [^Tensure src (last args)
        [^"[Lorg.nd4j.linalg.indexing.INDArrayIndex;" selection-indices _] (get-range-selection-indices nd (drop-last args))]
    (.put (o nd) selection-indices (o src))
    nd))

(defn set-axis-range!
  "Like `set-range!` but sets a selection along only a single axis (sets everything along other axes).
  `selection` must be a valid `selection` argument to `select-range`, and `axis` must be a valid axis index."
  [^Tensure nd axis selection val]
  (->> (assoc (vec (repeat (rank nd) :all)) axis selection)
       (#(conj % val))
       (apply set-range! nd)))

(defn shift
  "Shifts the elements in `nd` by `shift-amount` elements along single dimension `dim` or by the number of
  elements specified in `shifts`, a vector of shift amounts for the axes at that index. Positive shifts are
  up/left, and negative shifts are down/right. For instance, `(shift nd 1 3)` shifts the columns of a matrix
  left by three elements; and `(shift nd [-1, 2])` shifts the rows of `nd` down by 1 and the columns of nd
  left by 2 (the first row and last two columns will be all zeros)."
  ([^Tensure nd dim shift-amount]
   (let [dim (if (array? dim) (scalar->number dim) dim)
         shift-amount (if (array? shift-amount) (scalar->number shift-amount) shift-amount)]
     (shift nd (concat (repeat dim 0) [shift-amount]))))
  ([^Tensure nd shifts]
   (let [shifts (if (array? shifts) (array->vector shifts) shifts)
         shifts-for-all-dims (concat shifts (repeat (- (rank nd) (count shifts)) 0))
         nd-shape (shape nd)
         shifts-size (u/zip shifts-for-all-dims nd-shape)]
     (when (> (count shifts) (rank nd))
       (u/throw-str "Shifts '" shifts "' provided to `shift` exceed the dimensionality of tensor '" nd "'."))
     (if-not (every? (fn [[^long shift size]]
                       (< (Math/abs shift) size))
                     shifts-size)
       (zeros nd-shape)
       (let [src-selection (map-indexed (fn [dim [^long shift size]]
                                          (cond (zero? shift) :all
                                                (neg? shift) [0 (+ size shift)]
                                                :else [(Math/abs shift) size]))
                                        shifts-size)
             target-selection (map-indexed (fn [dim [^long shift size]]
                                             (cond (zero? shift) :all
                                                   (neg? shift) [(Math/abs shift) size]
                                                   :else [0 (- size shift)]))
                                           shifts-size)
             src-view (apply select-range nd src-selection)]
         (apply set-range! (zeros nd-shape) (concat target-selection [src-view])))))))

(declare join-along)
; The broadcasting method used below (which uses `join-along` in a loop to repeat `src` along the
; broadcasted dimensions) is (surprisingly) faster than the other methods tested, including the Nd4j
; broadcasting op.
(defn- broadcast-with
  "Performs `(f target-view src)` for every view of `target` that is of the same shape as `src` and that
  would align with `src` were `src` to be broadcast to the shape of `target`. `f` operates directly on the
  underling nd4j arrays."
  [target src f]
  (if (= (shape target) (shape src))
    (f (o target) (o src))
    (let [src-shape (shape src)
          target-shape (shape target)
          target-rank (rank target)
          src-rank (rank src)
          ; This is just `src-shape` padded with ones to match the rank of `target`.
          tile-shape (->> (partition src-rank 1 target-shape)
                          reverse
                          (keep-indexed (fn [i target-subshape]
                                          (when (every? (fn [[t s]] (or (= t s) (= s 1)))
                                                        (u/zip target-subshape src-shape))
                                            (concat (repeat (- target-rank src-rank i) 1)
                                                    src-shape
                                                    (repeat i 1)))))
                          first)
          ; This is the number of times `src` must be repeated in each dimension to match `target`.
          tiling-factor (map / target-shape tile-shape)
          _ (when-not tile-shape
              (u/throw-str "Cannot broadcast tensor of shape '" (shape src) "' to shape '"
                           (shape target) "'."))
          broadcasted-src (reduce-kv (fn [result axis repeat-n]
                                       (apply join-along axis (repeat repeat-n result)))
                                     (reshape src tile-shape)
                                     (vec tiling-factor))]
      (f (o target) (o broadcasted-src))))
  target)

(defn broadcast-like
  "Broadcasts `src` into a new tensor that's the shape of `target`. Returns `nil` if `target` is smaaller
  than `src` and throws an exception if `src` is smaller than `target` but cannot be broadcast to the shape
  of `target`."
  [^Tensure src ^Tensure target]
  (when (>= (ecount target) (ecount src))
    (broadcast-with (zeros (shape target)) src #(.assign (->o %1) (->o %2)))))

(defn broadcast
  "Returns a new tensor produced by braodcasting `nd` to shape `shape`."
  [^Tensure nd shape]
  (broadcast-with (zeros shape) nd #(.assign (->o %1) (->o %2))))

(defn join-along
  "Returns a new tensor produced by concatenating tensors `nds` along dimension `dim`. `nds` must be the
  same shape except for `dim`."
  [dim & nds]
  (if-not (every? #(not (scalar? %)) nds)
    (u/throw-str "Cannot concatenate scalars."))
  (let [first-rank (shape (first nds))
        dim (if (array? dim)
              (scalar->number dim)
              dim)
        concat-dim (if (vec? (first nds))
                     (do (when (>= dim (rank (first nds)))
                           (u/throw-str "Cannot select dimension '" dim "' of a vector."))
                         1)
                     dim)]
    (->> (map #(o %) nds)
         into-array
         (Nd4j/concat concat-dim)
         (#(Tensure. % (rank (first nds)))))))

(defn partition-along
  "Splits `nd` along `axis` into chunks of `partition-size` spaced by `step-size`. The last chunk is
  included even if it is not `partition-size`; this is similar to `clojure.core/partition-all` but for
  tensors rather than arbitrary collections. If not provided, `axis` defaults to `0`, `partition-size`
  defaults to `1`, and `step-size` defaults to `partition-size`. Returns a seq of tensors."
  ([^Tensure nd]
   (partition-along nd 0))
  ([^Tensure nd axis]
   (partition-along nd axis 1))
  ([^Tensure nd axis partition-size]
   (partition-along nd axis partition-size partition-size))
  ([^Tensure nd axis partition-size step-size]
   (let [partitioned-axis-size (dimension-count nd axis)]
     (->> (range 0 partitioned-axis-size step-size)
          (map (fn [partition-start-index]
                 (select-axis-range nd axis [partition-start-index
                                             (clojure.core/min (+ partition-start-index partition-size)
                                                               partitioned-axis-size)])))))))

; TODO: This is functionally identical to slices. Figure out which algorithm to keep and then get rid of this.
(defn split
  "Splits `nd` along `axis`, returning a seq of tensors with size 1 along `axis` and all other axes the same
  size as `nd`. Equivalent to `(partition-along nd axis 1 1)`."
  ([^Tensure nd]
   (split nd 0))
  ([^Tensure nd axis]
   (->> (dimension-count nd axis)
        (range 0)
        u/unchunk
        (pmap #(select-axis-range nd axis %)))))

(defn add-dimension
  "Reshapes `nd` to have an extra dimension of size 1 for axis index `axis`."
  ([^Tensure nd]
   (add-dimension nd 0))
  ([^Tensure nd axis]
   (let [[leading trailing] (split-at axis (shape nd))]
     (reshape nd (concat leading [1] trailing)))))

; TODO: Check axis in this function and above and throw informative error message if it's not a valid axis
; value or if tensors have incompatible dimensions/shapes
(defn stack
  "Given a series of tensors and, optionally, an axis index, returns a new tensor produced by 'stacking'
  the tensors along that axis. The axis index can be specified as the first argument but otherwise defaults
  to 0:
    (stack nd-a nd-b nd-c ...)
    (stack axis nd-a nd-b nd-c ...)
  The input tensors must have identical shapes. The returned tensor will be the same shape as the input
  tensors but will have an extra dimension at index `axis` of size `(count nds)`."
  [& args]
  (let [[axis nds] (if (array? (first args))
                     [0 args]
                     [(first args) (rest args)])]
    (->> (map #(add-dimension % axis) nds)
         (apply join-along axis))))

(defn rows
  "If `nd` is a tensor, returns a seq of views of vectors along `nd`'s innermost dimension. If `nd` is a
  matrix, this corresponds to a seq of row vectors. Throws an exception if `nd` is a scalar."
  [^Tensure nd]
  (if (scalar? nd)
    (u/throw-str "Can't get rows of a scalar.")
    (let [ond (o nd)
          row-dims (int-array [(dec (.rank ond))])
          row-count (.tensorssAlongDimension ond row-dims)]
      (pmap
        #(Tensure.
           (.tensorAlongDimension ond % row-dims)
           1)
        (range row-count)))))

(defn columns
  "If `nd` is a tensor, returns a seq of views of vectors along `nd`'s second-to-innermost dimension. If `nd`
  is a matrix, this corresponds to a seq of column vectors. Returns a seq of scalars if `nd` is a vector.
  Throws an exception if `nd` is a scalar."
  [^Tensure nd]
  (if (scalar? nd)
    (u/throw-str "Can't get columns of a scalar.")
    (let [ond (o nd)
          row-dims (int-array [(- (.rank ond) 2)])
          row-count (.tensorssAlongDimension ond row-dims)
          result-rank (if (vec? nd) 0 1)]
      (pmap
        #(Tensure.
           (.tensorAlongDimension ond % row-dims)
           result-rank)
        (range row-count)))))

(defn slices
  "Returns a seq of slice views through `nd` along the indicated `dimension` (defaults to the first dimension)."
  ([^Tensure nd]
   (slices nd 0))
  ([^Tensure nd dimension]
   (if (scalar? nd)
     (u/throw-str "Can't slice a scalar.")
     (let [ond (o nd)
           slice-rank (dec (rank nd))
           slice-dim (if (vec? nd) 1 dimension)
           get-slice (if (= slice-rank 1)
                       (fn [i]
                         (let [i-slice (.slice ond i slice-dim)]
                           (.reshape i-slice (int-array [1 (.length i-slice)]))))
                       (fn [i]
                         (.slice ond i slice-dim)))]
       (pmap
         #(Tensure.
            (get-slice %)
            slice-rank)
         (range (dimension-count nd dimension)))))))

(defn submatrix
  "Returns a view of a subregion of tensor `nd`. The result will have the same dimensionality as `nd`. The
  subregion can be specified as:
  - `row-start`, `row-length`, `col-start`, `col-length`, where `row-start` and `col-start` are the starting
    indices for the view in the 0th and 1st dimensions of `nd`, respectively, and `row-length` and `col-length`
    are the sizes of the subregion in its 0th and 1st dimension, respectively
  - `dimension`, `index-ranges`, where `dimension` is a dimension index and `index-range` is a two element
    vector like [`start-index`, `length`] that describes where the subregion should start and how long it
    should be in that dimension (the subregion will include the full length of all other dimensions)
  - `index-ranges`, a seq where each element is either a vector like [`start-index`, `length`] or `nil`
    (indicating the entire length of the dimension should be included. If the length of the seq is less than
    the dimensionality of `nd`, the entirety of trailing dimensions is included."
  ([^Tensure nd dimension index-range]
   (submatrix nd (concat (repeat dimension nil)
                         [index-range])))
  ([^Tensure nd row-start row-length col-start col-length]
   (submatrix nd [[row-start row-length] [col-start col-length]]))
  ([^Tensure nd index-ranges]
   (when-not (and (sequential? index-ranges)
                  (every? #(or (nil? %) (and (sequential? %)
                                             (= 2 (count %))
                                             (int? (first %))
                                             (>= (first %) 0)
                                             (pos-int? (second %))))
                          index-ranges))
     (u/throw-str "Invalid index-ranges provided to `submatrix`: " index-ranges ". `index-ranges` must be "
                  " a seq of [start length] pairs."))
   (apply select-range nd
          (map (fn [[start length]]
                 (if (and start length)
                   [start (+ start length)]
                   :all))
               (concat index-ranges
                       (repeat (- (rank nd) (count index-ranges)) nil))))))

; The following functions (mul-scalar, add-scalar, ...) are the equivalent of the Nd4j copy methods mul,
; add, ..., which return incorrect results on some views. For instance,
; (outer-product (array [4 5]) (array [1 2 3]) (array 2)) will give an incorrect result due to this.
; May be related to https://github.com/deeplearning4j/deeplearning4j/issues/7263.
(defn- mul-scalar
  "Returns a new tensor that's the product of tensor `nd` and `scalar`."
  [^Tensure nd ^Tensure scalar]
  (Tensure.
    (.muli (.dup (o nd)) (o scalar))
    (clojure.core/max (rank nd) (rank scalar))))

(defn- add-scalar
  "Returns a new tensor that's the sum of tensor `nd` and `scalar`."
  [^Tensure nd ^Tensure scalar]
  (Tensure.
    (.addi (.dup (o nd)) (o scalar))
    (clojure.core/max (rank nd) (rank scalar))))

(defn- sub-scalar
  "Returns a new tensor that's the difference of tensor `nd` and `scalar`."
  [^Tensure nd ^Tensure scalar]
  (Tensure.
    (.subi (.dup (o nd)) (o scalar))
    (clojure.core/max (rank nd) (rank scalar))))

(defn- div-scalar
  "Returns a new tensor that's the quotient of tensor `nd` and `scalar`."
  [^Tensure nd ^Tensure scalar]
  (Tensure.
    (.divi (.dup (o nd)) (o scalar))
    (clojure.core/max (rank nd) (rank scalar))))

(defn- rsub-scalar
  "Returns a new tensor that's the right difference of tensor `nd` and `scalar`."
  [^Tensure nd ^Tensure scalar]
  (Tensure.
    (.rsubi (.dup (o nd)) (o scalar))
    (clojure.core/max (rank nd) (rank scalar))))

(defn- rdiv-scalar
  "Returns a new tensor that's the right quotient of tensor `nd` and `scalar`."
  [^Tensure nd ^Tensure scalar]
  (Tensure.
    (.rdivi (.dup (o nd)) (o scalar))
    (clojure.core/max (rank nd) (rank scalar))))

(defn- eq-scalar
  "Returns a new boolean tensor representing the elementwise equality of tensor `nd` and `scalar`."
  [^Tensure nd ^Tensure scalar]
  (Tensure.
    ; We extract the Float from the Nd4j scalar, because Nd4j gives an exception when calling .eqi directly
    ; on the scalar.
    (.eqi (.dup (o nd)) (scalar->number scalar))
    (clojure.core/max (rank nd) (rank scalar))))

(defn- neq-scalar
  "Returns a new boolean tensor representing (not= nd-element `scalar`) for every element of `nd`."
  [^Tensure nd ^Tensure scalar]
  (Tensure.
    (.neqi (.dup (o nd)) (scalar->number scalar))
    (clojure.core/max (rank nd) (rank scalar))))

(defn- gt-scalar
  "Returns a new boolean tensor representing (> nd-element `scalar`) for every element of `nd`."
  [^Tensure nd ^Tensure scalar]
  (Tensure.
    (.gti (.dup (o nd)) (scalar->number scalar))
    (clojure.core/max (rank nd) (rank scalar))))

(defn- lt-scalar
  "Returns a new boolean tensor representing (< nd-element `scalar`) for every element of `nd`."
  [^Tensure nd ^Tensure scalar]
  (Tensure.
    (.lti (.dup (o nd)) (scalar->number scalar))
    (clojure.core/max (rank nd) (rank scalar))))

(defn- gte-scalar
  "Returns a new boolean tensor representing (>= nd-element `scalar`) for every element of `nd`."
  [^Tensure nd ^Tensure scalar]
  (Tensure.
    (.gtei (.dup (o nd)) (scalar->number scalar))
    (clojure.core/max (rank nd) (rank scalar))))

(defn- lte-scalar
  "Returns a new boolean tensor representing (<= nd-element `scalar`) for every element of `nd`."
  [^Tensure nd ^Tensure scalar]
  (Tensure.
    (.ltei (.dup (o nd)) (scalar->number scalar))
    (clojure.core/max (rank nd) (rank scalar))))

(defn mul!
  "Performs elementwise multiplication of tensor or scalar arguments, with implicit broadcasting, and places
  the result in the first argument."
  ([^Tensure a]
   (when (scalar? a)
     (u/throw-str "Cannot perform mutable operation on a scalar."))
   a)
  ([^Tensure a & more]
   (when (scalar? a)
     (u/throw-str "Cannot perform mutable operation on a scalar."))
   (doseq [b more]
     (if (scalar? b)
       (.muli (o a) (o b))
       (broadcast-with a b #(.muli (->o %1) (->o %2)))))
   a))

(defn mul
  "Performs elementwise multiplication of tensor or scalar arguments, with implicit broadcasting, and returns
  a new tensor with the result."
  ([]
   (array 1))
  ([^Tensure a]
   a)
  ([^Tensure a ^Tensure b & more]
   (let [result (cond (scalar? a) (mul-scalar b a)
                      (scalar? b) (mul-scalar a b)
                      :else (or (when-let [result (broadcast-like a b)]
                                  (.muli (o result) (o b))
                                  result)
                                (when-let [result (broadcast-like b a)]
                                  (.muli (o result) (o a))
                                  result)))]
     (reduce #(if (or (scalar? %1)
                      (< (ecount %1) (ecount %2)))
                (mul %1 %2)
                (mul! %1 %2))
             result
             more))))

(defn add!
  "Performs elementwise addition of tensor or scalar arguments, with implicit broadcasting, and places
  the result in the first argument."
  ([^Tensure a]
   (when (scalar? a)
     (u/throw-str "Cannot perform mutable operation on a scalar."))
   a)
  ([^Tensure a & more]
   (when (scalar? a)
     (u/throw-str "Cannot perform mutable operation on a scalar."))
   (doseq [b more]
     (if (scalar? b)
       (.addi (o a) (o b))
       (broadcast-with a b #(.addi (->o %1) (->o %2)))))
   a))

(defn add
  "Performs elementwise addition of tensor or scalar arguments, with implicit broadcasting, and returns
  a new tensor with the result."
  ([]
   (array 0))
  ([^Tensure a]
   a)
  ([^Tensure a ^Tensure b & more]
   (let [result (cond (scalar? a) (add-scalar b a)
                      (scalar? b) (add-scalar a b)
                      :else (or (when-let [result (broadcast-like a b)]
                                  (.addi (o result) (o b))
                                  result)
                                (when-let [result (broadcast-like b a)]
                                  (.addi (o result) (o a))
                                  result)))]
     (reduce #(if (or (scalar? %1)
                      (< (ecount %1) (ecount %2)))
                (add %1 %2)
                (add! %1 %2))
             result
             more))))

(defn sub!
  "Performs elementwise subtraction of tensor or scalar arguments, with implicit broadcasting, and places
  the result in the first argument."
  ([^Tensure a]
   (when (scalar? a)
     (u/throw-str "Cannot perform mutable operation on a scalar."))
   (.negi (o a))
   a)
  ([^Tensure a & more]
   (when (scalar? a)
     (u/throw-str "Cannot perform mutable operation on a scalar."))
   (doseq [b more]
     (if (scalar? b)
       (.subi (o a) (o b))
       (broadcast-with a b #(.subi (->o %1) (->o %2)))))
   a))

(defn negate
  "Returns the element-wise negation of `nd`."
  [^Tensure nd]
  (Tensure.
    (.neg (o nd))
    (rank nd)))

(defn sub
  "Performs elementwise subtraction of tensor or scalar arguments, with implicit broadcasting, and returns
  a new tensor with the result."
  ([]
   (array 0))
  ([^Tensure a]
   (negate a))
  ([^Tensure a ^Tensure b & more]
   (let [result (cond (scalar? b) (sub-scalar a b)
                      (scalar? a) (rsub-scalar b a)
                      :else (or (when-let [result (broadcast-like a b)]
                                  (.subi (o result) (o b))
                                  result)
                                (when-let [result (broadcast-like b a)]
                                  (.rsubi (o result) (o a))
                                  result)))]
     (reduce #(if (or (scalar? %1)
                      (< (ecount %1) (ecount %2)))
                (sub %1 %2)
                (sub! %1 %2))
             result
             more))))

(defn div!
  "Performs elementwise division of tensor or scalar arguments, with implicit broadcasting, and places
  the result in the first argument."
  ([^Tensure a]
   (when (scalar? a)
     (u/throw-str "Cannot perform mutable operation on a scalar."))
   (.rdivi (o a) ^java.lang.Float (float 1))
   a)
  ([^Tensure a & more]
   (when (scalar? a)
     (u/throw-str "Cannot perform mutable operation on a scalar."))
   (doseq [b more]
     (if (scalar? b)
       (.divi (o a) (o b))
       (broadcast-with a b #(.divi (->o %1) (->o %2)))))
   a))

(defn div
  "Performs elementwise division of tensor or scalar arguments, with implicit broadcasting, and returns
  a new tensor with the result."
  ([^Tensure a]
   (Tensure.
     (.rdiv (o a) ^java.lang.Float (float 1))
     (rank a)))
  ([^Tensure a ^Tensure b & more]
   (let [result (cond (scalar? b) (div-scalar a b)
                      (scalar? a) (rdiv-scalar b a)
                      :else (or (when-let [result (broadcast-like a b)]
                                  (.divi (o result) (o b))
                                  result)
                                (when-let [result (broadcast-like b a)]
                                  (.rdivi (o result) (o a))
                                  result)))]
     (reduce #(if (or (scalar? %1)
                      (< (ecount %1) (ecount %2)))
                (div %1 %2)
                (div! %1 %2))
             result
             more))))

(defn pow!
  "Computes the elementwise power function a^more[0]^more[1]... modifying `a` in place with the results."
  ([^Tensure a]
   a)
  ([^Tensure a & more]
   (doseq [b more]
     (if (scalar? b)
       (Transforms/pow (o a) (o b) false)
       (broadcast-with a b #(Transforms/pow (->o %1) (->o %2) false))))
   a))

(defn pow
  "Computes the elementwise power function a^more[0]^more[1]... and returns a new tensor with the result."
  ([^Tensure a]
   (clone a))
  ([^Tensure a ^Tensure b & more]
   (let [result (cond (scalar? b) (Tensure.
                                    (Transforms/pow (o a) (o b) true)
                                    (rank a))
                      (scalar? a) (let [result (filled (shape b) (scalar->number a))]
                                    (Transforms/pow (o result) (o b) false)
                                    result)
                      :else (or (when-let [result (broadcast-like a b)]
                                  (Transforms/pow (o result) (o b) false)
                                  result)
                                (Tensure.
                                  (Transforms/pow (o a) (o (broadcast-like b a)) true)
                                  (rank a))))]
     (reduce #(if (or (scalar? %1)
                      (< (ecount %1) (ecount %2)))
                (pow %1 %2)
                (pow! %1 %2))
             result
             more))))

(defn eq!
  "Calculates elementwise equality of tensor or scalar arguments, with implicit broadcasting, and places
  the result in the first argument."
  ([^Tensure a]
   (when (scalar? a)
     (u/throw-str "Cannot perform mutable operation on a scalar."))
   (fill! a 1))
  ([^Tensure a ^Tensure b]
   (when (scalar? a)
     (u/throw-str "Cannot perform mutable operation on a scalar."))
   (if (scalar? b)
     (.eqi (o a) (scalar->number b))
     (broadcast-with a b #(.eqi (->o %1) (->o %2))))
   a))

(defn eq
  "Calculates elementwise equality of tensor or scalar arguments, with implicit broadcasting, and places the
  result in the first argument."
  ([^Tensure a]
   (filled (shape a) 1))
  ([^Tensure a ^Tensure b]
   (cond (scalar? b) (eq-scalar a b)
         (scalar? a) (eq-scalar b a)
         :else (or (when-let [result (broadcast-like a b)]
                     (.eqi (o result) (o b))
                     result)
                   (when-let [result (broadcast-like b a)]
                     (.eqi (o result) (o a))
                     result)))))

(defn ne!
  "Calculates elementwise inequality of tensor or scalar arguments, with implicit broadcasting, and places
  the result in the first argument."
  ([^Tensure a]
   (when (scalar? a)
     (u/throw-str "Cannot perform mutable operation on a scalar."))
   (fill! a 0))
  ([^Tensure a ^Tensure b]
   (when (scalar? a)
     (u/throw-str "Cannot perform mutable operation on a scalar."))
   (if (scalar? b)
     (.neqi (o a) (scalar->number b))
     (broadcast-with a b #(.neqi (->o %1) (->o %2))))
   a))

(defn ne
  "Calculates elementwise inequality of tensor or scalar arguments, with implicit broadcasting, and places
  the result in the first argument."
  ([^Tensure a]
   (filled (shape a) 0))
  ([^Tensure a ^Tensure b]
   (cond (scalar? b) (neq-scalar a b)
         (scalar? a) (neq-scalar b a)
         :else (or (when-let [result (broadcast-like a b)]
                     (.neqi (o result) (o b))
                     result)
                   (when-let [result (broadcast-like b a)]
                     (.neqi (o result) (o a))
                     result)))))

(defn gt!
  "Calculates elementwise > of tensor or scalar arguments, with implicit broadcasting, and places the result
  in the first argument."
  ([^Tensure a]
   (when (scalar? a)
     (u/throw-str "Cannot perform mutable operation on a scalar."))
   (fill! a 1))
  ([^Tensure a ^Tensure b]
   (when (scalar? a)
     (u/throw-str "Cannot perform mutable operation on a scalar."))
   (if (scalar? b)
     (.gti (o a) ^java.lang.Number (.getFloat (o b) (int-array [0])))
     (broadcast-with a b #(.gti (->o %1) (->o %2))))
   a))

(defn gt
  "Returns a new tensor containing the elementwise result of (> a b), with implicit broadcasting."
  ([^Tensure a]
   (filled (shape a) 1))
  ([^Tensure a ^Tensure b]
   (cond (scalar? b) (gt-scalar a b)
         (scalar? a) (lt-scalar b a)
         :else (or (when-let [result (broadcast-like a b)]
                     (.gti (o result) (o b))
                     result)
                   (when-let [result (broadcast-like b a)]
                     (Tensure.
                       (.gt (o a) (o result))
                       (rank result)))))))

(defn lt!
  "Calculates elementwise < of tensor or scalar arguments, with implicit broadcasting, and places the result
  in the first argument."
  ([^Tensure a]
   (when (scalar? a)
     (u/throw-str "Cannot perform mutable operation on a scalar."))
   (fill! a 1))
  ([^Tensure a ^Tensure b]
   (when (scalar? a)
     (u/throw-str "Cannot perform mutable operation on a scalar."))
   (if (scalar? b)
     (.lti (o a) (scalar->number b))
     (broadcast-with a b #(.lti (->o %1) (->o %2))))
   a))

(defn lt
  "Returns a new tensor containing the elementwise result of (< a b), with implicit broadcasting."
  ([^Tensure a]
   (filled (shape a) 1))
  ([^Tensure a ^Tensure b]
   (cond (scalar? b) (lt-scalar a b)
         (scalar? a) (gt-scalar b a)
         :else (or (when-let [result (broadcast-like a b)]
                     (.lti (o result) (o b))
                     result)
                   (when-let [result (broadcast-like b a)]
                     (Tensure.
                       (.lt (o a) (o result))
                       (rank result)))))))

(defn le!
  "Calculates elementwise <= of tensor or scalar arguments, with implicit broadcasting, and places the result
  in the first argument."
  ([^Tensure a]
   (when (scalar? a)
     (u/throw-str "Cannot perform mutable operation on a scalar."))
   (fill! a 1))
  ([^Tensure a ^Tensure b]
   (when (scalar? a)
     (u/throw-str "Cannot perform mutable operation on a scalar."))
   (if (scalar? b)
     (.ltei (o a) (scalar->number b))
     (do (gt! a b)
         (eq! a (array 0))))
   a))

(defn le
  "Returns a new tensor containing the elementwise result of (<= a b), with implicit broadcasting."
  ([^Tensure a]
   (filled (shape a) 1))
  ([^Tensure a ^Tensure b]
   (cond (scalar? b) (lte-scalar a b)
         (scalar? a) (gte-scalar b a)
         :else (or (when-let [result (broadcast-like a b)]
                     (le! result b)
                     result)
                   (when-let [result (broadcast-like b a)]
                     (lt! result a)
                     (eq! result (array 0))
                     result)))))

(defn ge!
  "Calculates elementwise >= of tensor or scalar arguments, with implicit broadcasting, and places the result
  in the first argument."
  ([^Tensure a]
   (when (scalar? a)
     (u/throw-str "Cannot perform mutable operation on a scalar."))
   (fill! a 1))
  ([^Tensure a ^Tensure b]
   (when (scalar? a)
     (u/throw-str "Cannot perform mutable operation on a scalar."))
   (if (scalar? b)
     (.gtei (o a) (scalar->number b))
     (do (lt! a b)
         (eq! a (array 0))))
   a))

(defn ge
  "Returns a new tensor containing the elementwise result of (>= a b), with implicit broadcasting."
  ([^Tensure a]
   (filled (shape a) 1))
  ([^Tensure a ^Tensure b]
   (cond (scalar? b) (gte-scalar a b)
         (scalar? a) (lte-scalar b a)
         :else (or (when-let [result (broadcast-like a b)]
                     (ge! result b)
                     result)
                   (when-let [result (broadcast-like b a)]
                     (gt! result a)
                     (eq! result (array 0))
                     result)))))

(defn equals
  "Returns true iff every element of `a` is within +/- `eps` of the corresponding element in `b`, and
  false otherwise. `eps` defaults to 1e-6."
  ([]
   true)
  ([^Tensure a ^Tensure b]
   (and (= (shape a) (shape b))
        (.equals (o a) (o b))))
  ([^Tensure a ^Tensure b eps]
   (and (= (shape a) (shape b))
        (.equalsWithEps (o a) (o b) eps))))

(defn- int-array-2d
  "Returns a Java int[][] with data from `data`, a two-dimensional Clojure vector."
  [data]
  (if (seq data)
    (let [inner-arrays (mapv int-array data)
          outer-len (count inner-arrays)
          ^"[[I" outer-array (make-array (type (first inner-arrays)) outer-len)]
      (doseq [[i ^ints inner-array] (u/zip (range outer-len) inner-arrays)]
        (aset outer-array i inner-array))
      outer-array)
    (make-array Integer/TYPE 0)))

(defn- tensor-dot
  "Computes the tensor dot product between `a` and `b`. The result will be a new tensor produced by
  multiplying all matrices along the highest two dimensions of `a` by all matrices along the lowest two
  dimensions of `b`."
  [a b]
  (let [dims (int-array-2d [[(dec (.rank (o a)))] [0]])]
    (Tensure.
      (Nd4j/tensorMmul (o a) (o b) dims)
      (- (+ (rank a) (rank b)) 2))))

(defn mmul
  "Computes:
    - the inner product of `a` and `b` when `a` and `b` are matrices or vectors
    - the inner product of the highest dimension of the tensor with the vector when one argument is a tensor
      and the other is a vector
    - the tensor dot product of `a` and `b` when either `a` or `b` is a tensor and the other is a tensor
      or matrix
    - element-wise product when `a` or `b` is a scalar
  When a and b are tensors of shape [a b c d e] and [f g h i j], then the product will be a tensor of shape
  [a b c d g e h i j] - i.e. the result is produced by multiplying every d x e matrix from `a` by every
  f x g matrix from `b` to get a * b * c * h * i * j d x g products."
  ([]
   (array 1))
  ([^Tensure a]
   a)
  ([^Tensure a ^Tensure b]
   (if (or (scalar? a) (scalar? b))
     (mul a b)
     (let [b (if (vec? b)
               (if (and (vec? a) (not= (shape a) (shape b)))
                 (u/throw-str "Cannot multiple a vector of size '" (ecount a) "' by a vector of size '" (ecount b)
                              "'. Sizes must match.")
                 (Tensure.
                   (.transpose (o b))
                   1))
               b)
           ao (o a)
           bo (o b)
           result (->o (if (and (= (.rank ao) 2) (= (.rank bo) 2))
                         (.mmul ao bo)
                         (o (tensor-dot a b))))
           ; The result is an Nd4j array resulting from treating both arguments as matrices or tensors.
           ; If `a` is a vector, we need to drop the lowest dimension, and if `b` is a vetor, we need to
           ; drop the highest dimension. Nd4j correctly handles shapes with ranks < 2.
           new-shape (->> (if (vec? a)
                            (rest (.shape result))
                            (.shape result))
                          (#(if (vec? b)
                              (drop-last %)
                              %))
                          long-array)]
       (Tensure.
         (.reshape result new-shape)
         (count new-shape)))))
  ([^Tensure a ^Tensure b & more]
   (reduce mmul (mmul a b) more)))

; See https://github.com/deeplearning4j/nd4j/issues/1229
(defn outer-product
  "Returns the outer product of the tensor arguments."
  ([]
   (array 1))
  ([^Tensure a]
   a)
  ([^Tensure a ^Tensure b]
   (if (or (scalar? a) (scalar? b))
     (mul a b)
     (let [ao (o a)
           result (-> (Nd4j/repeat ao (ecount b))
                      (.reshape (int-array (concat [(ecount b)] (shape a))))
                      (.permutei (int-array (concat (range 1 (inc (rank a))) [0])))
                      (.reshape (long-array (concat (shape a) (shape b))))
                      (Tensure. (+ (rank a) (rank b))))]
       (broadcast-with result (reshape b (concat (repeat (rank a) 1) (shape b))) #(.muli (->o %1) (->o %2)))
       result)))
  ([^Tensure a ^Tensure b & more]
   (reduce outer-product (outer-product a b) more)))

(defn emax
  "Returns a new scalar that is the maximum element in `nd`."
  [^Tensure nd]
  (array (.maxNumber (o nd))))

(defn emin
  "Returns a new scalar that is the minimum element in `nd`."
  [^Tensure nd]
  (array (.minNumber (o nd))))

(defn esum
  "Returns a new scalar that is the sum of all elements in `nd`."
  [^Tensure nd]
  (array (.sumNumber (o nd))))

(defn- make-reducing-fn
  "Returns a function that reduces a tensor along some dimensions using `f`. `f` must take as arguments an
  nd4j array and an integer array of axis indices and return a new nd4j array with the result. The
  returned function will have the signatures `(f ^Tensure nd axes)` and `(f ^Tensure nd axes collapse)`,
  where `axes` is either a single axis or a vector of `axes`, and `collapse` is a boolean (default `true`)
  indicating whether or not to remove the reduced dimensions."
  [f]
  (fn reducer
    ([^Tensure nd axes]
     (reducer nd axes true))
    ([^Tensure nd axes collapse]
     (let [axes-set (if (number? axes)
                      #{axes}
                      (into #{} axes))
           _ (when-not (every? #(or (pos-int? %) (zero? %)) axes-set)
               (u/throw-str "Invalid axes: '" (vec axes-set) "'."))
           nd-rank (rank nd)
           nd-shape (shape nd)
           _ (when (some #(>= % nd-rank) axes-set)
               (u/throw-str "Cannot reduce a tensor of shape '" nd-shape "' along axes '" (vec axes-set) "'."))
           result-shape (if collapse
                          (keep-indexed (fn [i size]
                                          (when (not (axes-set i))
                                            size))
                                        nd-shape)
                          (map-indexed (fn [i size]
                                         (if (axes-set i)
                                           1
                                           size))
                                       nd-shape))
           reduced-axes (if (and (= 1 nd-rank)
                                 (= #{0} axes-set))
                          [1]
                          axes-set)]
       (if (seq reduced-axes)
         (-> (Tensure.
               (f (o nd) (int-array reduced-axes))
               (count result-shape))
             (reshape result-shape))
         nd)))))

(def sum-along
  "Returns a new tensor that's the result of summing `nd` along `axes`, either a single axis or a vector of
  axes. If `collapse` is `true` (the default), the summed dimensions are removed from the result; otherwise,
  the rank of the returned tensor will be the same as the rank of the input tensor, with the reduced axes
  having size 1."
  (make-reducing-fn #(.sum (->o %1) (->o %2))))

(def max-along
  "Returns a new tensor that's the result of finding the minimum of `nd` along `axes`, either a single axis
  or a vector of axes. If `collapse` is `true` (the default), the reduced dimensions are removed from the
  result; otherwise, the rank of the returned tensor will be the same as the rank of the input tensor, with
  the reduced axes having size 1."
  (make-reducing-fn #(.max (->o %1) (->o %2))))

(def min-along
  "Returns a new tensor that's the result of finding the minimum of `nd` along `axes`, either a single axis
  or a vector of axes. If `collapse` is `true` (the default), the reduced dimensions are removed from the
  result; otherwise, the rank of the returned tensor will be the same as the rank of the input tensor, with
  the reduced axes having size 1."
  (make-reducing-fn #(.min (->o %1) (->o %2))))

(defn- ->nd4j-axis
  "Given a number that's a valid axis index or a seq of axis indices, returns a Java Integer array of the
  axis(es). Throws an Exception if `axis` is not valid or if any axis is out-of-bounds for a tensor of
  rank `rank`."
  [axis rank]
  (let [axis-vec (->> (if (number? axis)
                        [axis]
                        (vec (distinct axis)))
                      (#(if (seq %)
                          %
                          (range rank))))]
    (when-not (every? #(and (or (pos-int? %) (zero? %))
                            (< % rank))
                      axis-vec)
      (u/throw-str "Invalid axes: '" axis-vec "'."))
    (if (and (= axis-vec [0]) (= rank 1))
      (int-array [1])
      (int-array axis-vec))))

(defn argmax-along
  "Given a tensor and either a single axis index or a seq of axis indices, returns a new tensor containing
  the index of the maximum element along the axis(es). When more than one axis is specified, the returned
  value is a linear index into a tensor where the indicated axes have been collapsed in row major order."
  [^Tensure nd axis]
  (let [axes (->nd4j-axis axis (rank nd))
        new-rank (- (rank nd) (count axes))
        result-o (.argMax (o nd) axes)]
    (Tensure.
      (if (= 1 new-rank)
        (.reshape result-o (int-array [(.length result-o)]))
        result-o)
      new-rank)))

(defn argmin-along
  "Given a tensor and either a single axis index or a seq of axis indices, returns a new tensor containing
  the index of the minimum element along the axis(es). When more than one axis is specified, the returned
  value is a linear index into a tensor where the indicated axes have been collapsed in row major order."
  [^Tensure nd axis]
  (let [axes (->nd4j-axis axis (rank nd))
        new-rank (- (rank nd) (count axes))
        result-o (.argMax (.neg (o nd)) axes)]
    (Tensure.
      (if (= 1 new-rank)
        (.reshape result-o (int-array [(.length result-o)]))
        result-o)
      new-rank)))

(defn emean
  "Returns a new scalar that is the mean of all elements in `nd`."
  [^Tensure nd]
  (array (.meanNumber (o nd))))

(defn estdev
  "Returns a new scalar that is the standard deviation of all elements in `nd`."
  [^Tensure nd]
  (array (.stdNumber (o nd))))

(defn max
  "Returns the elementwise max of tensors, broadcasting when necessary."
  ([^Tensure a]
   a)
  ([^Tensure a ^Tensure b]
   (if (and (scalar? a) (scalar? b))
     (array (Math/max (scalar->number a) (scalar->number b)))
     (let [result (ge a b)
           b-els (mul! (eq result (array 0)) b)]
       (add! (mul! result a) b-els))))
  ([^Tensure a ^Tensure b & more]
   (reduce max (max a b) more)))

(defn min
  "Returns the elementwise min of tensors, broadcasting when necessary."
  ([^Tensure a]
   a)
  ([^Tensure a ^Tensure b]
   (if (and (scalar? a) (scalar? b))
     (array (Math/min (scalar->number a) (scalar->number b)))
     (let [result (le a b)
           b-els (mul! (eq result (array 0)) b)]
       (add! (mul! result a) b-els))))
  ([^Tensure a ^Tensure b & more]
   (reduce min (min a b) more)))

(defn abs!
  "Computes the absolute values of all elements in `nd`, modifying `nd` in place with the results."
  [^Tensure nd]
  (Transforms/abs (o nd) false)
  nd)

(defn abs
  "Computes the absolute value of all elements in `nd`, and returns a new tensor with the result."
  [^Tensure nd]
  (Tensure.
    (Transforms/abs (o nd) true)
    (rank nd)))

(defn log!
  "Computes the natural logarithm of all elements in `nd`, modifying `nd` in place with the results."
  [^Tensure nd]
  (Transforms/log (o nd) false)
  nd)

(defn log
  "Computes the natural logarithm of all elements in `nd`, and returns a new tensor with the result."
  [^Tensure nd]
  (Tensure.
    (Transforms/log (o nd) true)
    (rank nd)))

(defn log10!
  "Computes the base 10 logarithm of all elements in `nd`, modifying `nd` in place with the results."
  [^Tensure nd]
  (Transforms/log (o nd) 10 false)
  nd)

(defn log10
  "Computes the base 10 logarithm of all elements in `nd`, and returns a new tensor with the result."
  [^Tensure nd]
  (Tensure.
    (Transforms/log (o nd) 10 true)
    (rank nd)))

(defn sqrt!
  "Computes the square root of all elements in `nd`, modifying `nd` in place with the results."
  [^Tensure nd]
  (Transforms/sqrt (o nd) false)
  nd)

(defn sqrt
  "Computes the square root of all elements in `nd`, and returns a new tensor with the result."
  [^Tensure nd]
  (Tensure.
    (Transforms/sqrt (o nd) true)
    (rank nd)))

(defn cos!
  "Computes the cosine of all elements in `nd`, modifying `nd` in place with the results."
  [^Tensure nd]
  (Transforms/cos (o nd) false)
  nd)

(defn cos
  "Computes the cosine of all elements in `nd`, and returns a new tensor with the result."
  [^Tensure nd]
  (Tensure.
    (Transforms/cos (o nd) true)
    (rank nd)))

(defn sin!
  "Computes the sine of all elements in `nd`, modifying `nd` in place with the results."
  [^Tensure nd]
  (Transforms/sin (o nd) false)
  nd)

(defn sin
  "Computes the sine of all elements in `nd`, and returns a new tensor with the result."
  [^Tensure nd]
  (Tensure.
    (Transforms/sin (o nd) true)
    (rank nd)))

(defn tan!
  "Computes the tangent of all elements in `nd`, modifying `nd` in place with the results."
  [^Tensure nd]
  (Transforms/tan (o nd) false)
  nd)

(defn tan
  "Computes the tangent of all elements in `nd`, and returns a new tensor with the result."
  [^Tensure nd]
  (Tensure.
    (Transforms/tan (o nd) true)
    (rank nd)))

(defn acos!
  "Computes the arccosine of all elements in `nd`, modifying `nd` in place with the results."
  [^Tensure nd]
  (Transforms/acos (o nd) false)
  nd)

(defn acos
  "Computes the arcosine of all elements in `nd`, and returns a new tensor with the result."
  [^Tensure nd]
  (Tensure.
    (Transforms/acos (o nd) true)
    (rank nd)))

(defn asin!
  "Computes the arcsine of all elements in `nd`, modifying `nd` in place with the results."
  [^Tensure nd]
  (Transforms/asin (o nd) false)
  nd)

(defn asin
  "Computes the arcsine of all elements in `nd`, and returns a new tensor with the result."
  [^Tensure nd]
  (Tensure.
    (Transforms/asin (o nd) true)
    (rank nd)))

(defn atan!
  "Computes the arcsine of all elements in `nd`, modifying `nd` in place with the results."
  [^Tensure nd]
  (Transforms/atan (o nd) false)
  nd)

(defn atan
  "Computes the arcsine of all elements in `nd`, and returns a new tensor with the result."
  [^Tensure nd]
  (Tensure.
    (Transforms/atan (o nd) true)
    (rank nd)))

(defn tanh!
  "Computes the hyperbolic tangent of all elements in `nd`, modifying `nd` in place with the results."
  [^Tensure nd]
  (Transforms/tanh (o nd) false)
  nd)

(defn tanh
  "Computes the hyperbolic tangent of all elements in `nd`, and returns a new tensor with the result."
  [^Tensure nd]
  (Tensure.
    (Transforms/tanh (o nd) true)
    (rank nd)))

(defn cosh!
  "Computes the hyperbolic cosine of all elements in `nd`, modifying `nd` in place with the results."
  [^Tensure nd]
  (Transforms/cosh (o nd) false)
  nd)

(defn cosh
  "Computes the hyperbolic cosine of all elements in `nd`, and returns a new tensor with the result."
  [^Tensure nd]
  (Tensure.
    (Transforms/cosh (o nd) true)
    (rank nd)))

(defn sinh!
  "Computes the hyperbolic sine of all elements in `nd`, modifying `nd` in place with the results."
  [^Tensure nd]
  (Transforms/sinh (o nd) false)
  nd)

(defn sinh
  "Computes the hyperbolic sine of all elements in `nd`, and returns a new tensor with the result."
  [^Tensure nd]
  (Tensure.
    (Transforms/sinh (o nd) true)
    (rank nd)))

; TODOs:
; - Add test for:
;     - add-dimension
;     - stack
;     - select-axis-range
;     - set-axis-range
;     - partition-along
;     - split-along
;     - add-dimension
;     - split-along
;     - Basic math function log!, log, log10!, log10, sqrt!, sqrt, cos, cos!, sin, sin!, tan, tan!,
;       acos, acos!, asin, asin!, atan, atan!, cosh, cosh!, sinh, sinh!, tanh, tanh!
; - Add functions for:
;    - padding
;    - tiling
;    - bernouli distributions
; - Use Transforms functions (not, or, greaterThanOrEqual, etc) for comparison operators
; - Figure out how to get rid of pmap in broadcast-with
; - Technically should be able to perform mutable operations (e.g. add!) on scalars?
; - Support broadcasting and setting many values to a scalar in set-range!?
; - switch argument order of broadcast-like to match core.matrix and make public?
; - Remove essentially redundant functions to simplify the API (e.g. do we really need both `scalar->number`
;   and `array->vector`?
