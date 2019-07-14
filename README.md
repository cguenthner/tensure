# Tensure

Tensure is a Clojure library for working with N-dimensional single precision numeric arrays (tensors). It uses
[nd4j](https://github.com/deeplearning4j/nd4j) for most of its functionality but provides a Clojure-friendly
API based largely on [`core.matrix`](https://github.com/mikera/core.matrix) with some important
differences. Although Tensure should be useful for a wide range of numerical computing applications, it was
designed specifically to facilitate neural network model architecture research and has some limitations as a
result (in particular, see information on [Data types](#data-types) and [Hardware
architecture](#hardware-architecture).

## Getting Started

Add tensure to your dependencies. If using leiningen, add the following to your `:dependencies` in
`project.clj`:

```
[tensure "0.1.0"]
```

Require the `core` namespace (`m` as an alias is convention for most other Clojure matrix math libraries):

```
(require '[tensure.core :as m])
```

Consult the [API docs](https://cguenthner.github.io/tensure/docs/index.html) for details. Since Tensure uses nd4j on the backend, it may also be
helpful to check out the [nd4j overview](https://deeplearning4j.org/docs/latest/nd4j-overview). The remainder
of this overview assumes a basic knowledge of tensor operations. Note that most of the examples below operate
on vectors and matrices for simplicitly but that most operations generalize to higher dimensional arrays.

## Creating tensors

There are several ways to create `Tensure`s:

1. **From Clojure data structures** using `array`:

    ```
    (m/array [[1 2 3] [4 5 6]])
    ;; Creates a 2 x 3 matrix:
    ;; => #Tensure
    ;;    [[    1.0000,    2.0000,    3.0000],
    ;;     [    4.0000,    5.0000,    6.0000]]
    ```

    There are a few more specialized functions for constructing tensors of particular dimensionality: `scalar`
    and `scalar-array` (these are aliases for the same function) for constructing a scalar from a Java number
    and `matrix` for constructing a 2-dimensional tensor from a Clojure data structure. These differ from
    `array` only in that they will throw `Exception`s if the input data is not of the expected dimensionality.

2. **_Ex nihilo_, with elements drawn from a variety of distributions**:

    ```
    (m/zeros [2 3])
    ;; Creates a 2 x 3 matrix (2-dimensional tensor) filled with 0's:
    ;; => #Tensure
    ;;    [[         0,         0,         0],
    ;;     [         0,         0,         0]]

    (m/ones [3])
    ;; Creates a 3-element vector (1-dimensional tensor) filled with 1's:
    ;; => #Tensure
    ;;    [[    1.0000,    1.0000,    1.0000]]

    (m/filled [3 2 2] 7)
    ;; Creates a 3 x 2 x 2 tensor filled with 7's:
    ;; => #Tensure
    ;;    [[[    7.0000,    7.0000],
    ;;      [    7.0000,    7.0000]],
    ;;     [[    7.0000,    7.0000],
    ;;      [    7.0000,    7.0000]],
    ;;     [[    7.0000,    7.0000],
    ;;      [    7.0000,    7.0000]]]

    (m/filled nil Math/PI)
    ;; Creates a scalar (0-dimensional tensor) equal to the specified number.
    ;; `nil` is the shape of a scalar.
    ;; => #Tensure
    ;;    3.1416

    (m/sample-rand-int [1 4] 7)
    ;; Creates a 1 x 4 matrix filled with elements drawn from a uniform random distribution of integers
    ;; over [0, 7).
    ;; => #Tensure
    ;;    [[    4.0000,    2.0000,    2.0000,    5.0000]]
    ;; Another call returns a different result.
    (m/sample-rand-int [1 4] 7)
    ;; => #Tensure
    ;;    [[    6.0000,    1.0000,    4.0000,    5.0000]]

    (m/sample-uniform [4 1])
    ;; Creates a 4 x 1 matrix drawn from a uniform distribution over [0, 1).
    ;; => #Tensure
    ;;    [0.1760,
    ;;     0.2056,
    ;;     0.9059,
    ;;     0.5968]

    (m/sample-normal [2 2])
    ;; Creates a 2 x 2 matrix drawn from a normal distribution with mean = 0 and standard deviation = 1.
    ;; => #Tensure
    ;;    [[    0.3159,    1.0631],
    ;;     [    1.6751,    1.3199]]

    ;; Functions that produce tensors filled with random numbers all take as an optional last argument a seed
    ;; for a shared random-number generator (RNG):
    (m/sample-rand-int [1 3] 10 0) ; => #Tensure [[    9.0000,    7.0000,    8.0000]]
    (m/sample-rand-int [1 3] 10 0) ; => #Tensure [[    9.0000,    7.0000,    8.0000]]
    ;; The above two function calls both return the same value, because the RNG is reset with seed = 0 before
    ;; each tensor is generated.

    ;; The RNG seed can also be reset globally:
    (m/set-rng-seed! 0)
    (m/sample-uniform [3]) ;; => #Tensure [    0.9968,    0.7879,    0.8322]
    (m/sample-normal [2]) ;; => #Tensure [   -1.2816,   -1.3209]
    (m/sample-rand-int nil 100) ;; => #Tensure 24.0000
    The above 4 function calls will always produce the same output.
    ```

3. **From other `Tensure`s using `clone`** (more on this below):

    ```
    (def a (m/array [1 2 3]))
    (def b (m/clone a))
    ;; a and b are distinct objects pointing to distinct underlying data
    ```

## Some general principles

### Views and mutation

In order to work effectively with tensors, it's necessary to understand some details of how tensors and
operations on them are implemented. A tensor essentially consists of a pointer to the data (an array of
floats) and some metadata about how those data are "organized" with respect to the tensor's axes. (See also
nd4j's documentation on [NDArrays: How Are They Stored In
Memory](https://deeplearning4j.org/docs/latest/nd4j-overview#inmemory).) Some operations on tensors can be
performed merely by changing the metatdata. For instance, reshaping a tensor (turning a 4-element vector into
a 2x2 matrix, for example) doesn't change the underlying data: it changes only the number of axes and/or the
size of the axes. Similarly, transposing a tensor doesn't changing the data it contains: it changes only the
arrangement of those data with respect to the tensor's axes. Reshaping and transposing could be achieved by
copying the original data into a new linear array in an order determined by the operation, and this would
require O(n) time, where n is the total number of elements. But the metadata can be changed in constant
time. Changing the metatdata--rather than copying the data itself--can greatly speed up many computations, so
this is how tensors are implemented in nd4j. However, this means that multiple tensors that _appear_ to be
different actually reference the same underlying data.

```
(def a (m/array [[1 2] [3 4]]))
;; => #Tensure
;;    [[    1.0000,    2.0000],
;;     [    3.0000,    4.0000]]

(def ta (m/transpose a))
;; => #Tensure
;;    [[    1.0000,    3.0000],
;;     [    2.0000,    4.0000]]

(def ra (m/reshape a [4]))
;; => #Tensure [[    1.0000,    2.0000,    3.0000,    4.0000]]

(m/same-data? a ta ra)
;; => true

(m/same-data? a (m/array [[1 2] [3 4]]))
;; => false

(m/same-data? a (m/clone ta))
;; => false
```

Given a series of tensor arguments, `same-data?` returns `true` iff they all share the same underlying
data. `ta` and `ra` _look_ like different tensors from `a`: they all have the same set of elements, but the
arrangement of those elements is different. Nonetheless, they all share the same underlying
data. Conceptually, this should not be foregin to Clojure programmers: Clojure's fast immutable data
structures are possible because they share data--every change to one of them does not result in the entire
data structure being copied. Structural sharing is not a problem when the data are fully immutable. The
problem arises when the data are mutable. Consider what happens when operating on these data:

```
(def b (m/array [[10 20] [30 40]]))
(def a+b (m/add a b))
;; => #Tensure
;;    [[   11.0000,   22.0000],
;;     [   33.0000,   44.0000]]

a ; evaluate a
;; => #Tensure
;;    [[    1.0000,    2.0000],
;;     [    3.0000,    4.0000]]
```

`add` returns a new tensor with new underlying data (`a` and `b` remain unchanged). No problem here. In
contrast, `add!` places the result in the first argument:

```
(m/add! a b)
a
;; => #Tensure
;;    [[   11.0000,   22.0000],
;;     [   33.0000,   44.0000]]

ta ; evaluate ta
;; => #Tensure
;;    [[   11.0000,   22.0000],
;;     [   33.0000,   44.0000]]

ra ; evaluate ra
:; #Tensure [[   11.0000,   22.0000,   33.0000,   44.0000]]

b ; evaluate b
;; => #Tensure
;;    [[   10.0000,   20.0000],
;;     [   30.0000,   40.0000]]
;; b remains unchanged; only the first argument is mutated
```

This is exactly the type of problem Clojure was designed to avoid: we can have a piece of code (like `(m/add!
a b)`) that _looks_ like it's at most modifying `a`, but since `a`, `ta`, and `ra` share data, they are all
modified. In general, _if you never use a Tensure function with `!` you can think of tensors as immutable_ and
avoid this complication entirely. However, tensor operations are frequently performance bottlencks, and
sometimes the complication of introducing mutability is acceptable for the sake of speed. For example:

```
(def a (m/sample-uniform [1e4 1e4]))
(def b (m/sample-uniform [1e4 1e4]))

(def mutation-result (time (-> (m/div a b)
                               (m/sub! b)
                               (m/pow! (m/array 2))
                               (m/mul! a)
                               m/abs!)))
;; "Elapsed time: 1019.628086 msecs"

(def copy-result (time (-> (m/div a b)
                           (m/sub b)
                           (m/pow (m/array 2))
                           (m/mul a)
                           m/abs)))
;; "Elapsed time: 2070.693558 msecs"

(m/equals copy-result mutation-result)
 ;; => true
```

The calculation used for `mutation-result` performs a division and places the result in a new tensor; it then
uses a series of mutable operations to modify this result. The calculation used for `copy-result` copies the
data at every step of the computation. The two calculations produce the same result, but the former is about
twice as fast in this quick-and-dirty benchmark. The mutable operations (those suffixed with `!`) are provided
for cases where they can improve performance over the copying version of the operation. If you do need to use
functions with `!` for performance reasons, then it's important to remember that other tensors could be
referencing the same underlying data. Note, however, that the mutating operations will not always be
meaningfully faster. Whether they are or not depends on the amount of the data, the nature of the operations,
and the architecture executing the computation. Regardelss, you should think of tensors as "views" over some
data and clearly distinguish between the "view" and the data.

To further complicate matters, some operations, given an input tensor, can return a new view over the same
data _or_ copy the data and return a tensor referencing that copied data:

```
(m/same-data? a (m/transpose a))
;; => true

(m/same-data? a (m/transpose (m/transpose a)))
;; => true

(m/same-data? a (m/reshape a [4]))
;; => true

(m/same-data? a (m/transpose (m/reshape a [4])))
;; => true

(m/same-data? a (m/reshape (m/transpose a) [4]))
;; => false
```

Note that the result of reshaping and then transposing `a` is a view over the original data of `a`, while the
result of transposing and then reshaping `a` references a different underlying data structure from `a`.

### Equality semantics

Two tensors, a and b, are equal if: 1) they are the same shape, and 2) for every possible element index (i, j,
... ), a[i, j, ...] == b[i, j, ...], where == is standard numerical equality. Thus:

```
(def a (m/array [1 2 3]))

(= a (m/array [1 2 3]))
;; => true

(= a (m/add (m/array [0 1 2])
            (m/ones [3])))
;; => true

(= a (m/transpose a))
;; => true
;; Since `a` is a vector, `a` and `(m/transpose a)` both have shape [3] with elements [1 2 3]. Tensure does
;; not distinguish between column and row vectors. On the other hand:

(m/equals a (m/reshape a [1 3]))
;; => false
;; `a` is a vector, and `(m/reshape a [1 3])` is a matrix. The former has shape [3] and the latter shape [1 3].
;; Their dimensionalities are different.

(m/equals (m/array [[1 2 3]]) (m/array [[1] [2] [3]]))
;; => false
;; Row and column matrices are different.
```

Equality can _appear_ to break down only when mutation is introduced:

```
(= a (m/add! a (m/ones [3])))
;; => true
```

The above example _looks_ like it should evaluate to `false`: 1 + x != x. But since `(m/add! a (m/ones [3]))`
is evaluated prior to evaluation of the equality, `a` is updated with the result of the addition before the
equality is evaluated.

### Data types

Currently, all Tensure data are stored as single-precision floating point values. nd4j supports
double-precision floating point arrays, and modifying Tensure to support them should be relatively simple. If
you need double precision, consider [contributing](#contributing).

### Hardware architecture

Currently, Tensure uses nd4j's CPU backend. nd4j also has a backend for NVIDIA CUDA-compatible GPUs, and
updating Tensure to use CUDA should (theoretically) be trivial. If you'd like to run Tensure on a GPU (or on
multiple GPU's), consider [contributing](#contributing).

## Getting information about tensors

The following table gives an overview of the functions Tensure provides for getting information about
tensors. Note that dimensions/axes are identified by indices starting at 0.

| Function          | Returns                                                    |
| ----------------- | ---------------------------------------------------------- |
| `array?`          | `true` iff the argument is a Tensure tensor                |
| `rank` / `dimensionality` | Number of dimensions in the tensor                 |
| `shape`           | shape of the tensor as a vector of integers                |
| `scalar?`         | `true` iff the argument is a 0-dimensional tensor (scalar) |
| `vec?`            | `true` iff the argument is a 1-dimensional tensor (vector) |
| `matrix?`         | `true` iff the argument is a 2-dimensional tensor (matrix) |
| `dimension-count` | `(dimension-count tensor dimension-index)` returns the size of the dimension indicated by `dimension-index` in `tensor` |
| `row-count`       | Size of the first dimension (index 0); same as `(dimension-count tensor 0)` |
| `column-count`    | Size of the second dimension (index 1); same as `(dimension-count tensor 1)` |
| `ecount`          | Total number of elements in a tensor                       |

## Conversion between tensors and Java/Clojure types

The section on [Creating tensors](#creating-tensors) describes how to convert Clojure data structures and
numbers into Tensure tensors. There are also a handful of functions for converting Tensure tensors into
Clojure/Java data types:

| Function          | Description                                                          |
| ----------------- | -------------------------------------------------------------------- |
| `sclar->number`   | Converts a Tensure scalar into a `java.lang.Float`                   |
| `->number`        | Converts a Java number _or_ a Tensure scalar into a Java number      |
| `->int`           | Like `->number` but coerces the result into an integer               |
| `eseq`            | Returns a Clojure seq of the elements of a tensor in row-major order |
| `array->vector`   | Converts a Tensure tensor into a `clojure.lang.PersistentVector` representation (e.g. a vector of vectors for a matrix) |
|

## Tensor operations

Tensure includes many functions for operating on tensors. There are few general points that are useful to keep in mind while learning the API:
- The API is very similar to [`core.matrix`](https://github.com/mikera/core.matrix), but Tensure does not
  (yet) implement the `core.matrix` interfaces. Moreover, Tensure does not have equivalents of all
  `core.matrix` functions, and it has some functions that do not exist in `core.matrix`.
- Operations typically only work on tensors. For example, `(m/add (m/array [1 2 3]) (m/array 2))` is valid,
  but `(m/add (m/array [1 2 3]) 2)` will throw an exception, because `2` (a `java.lang.Long`) is not an
  acceptable argument to `m/add`. This requirement provides consistency and also avoids silent performance
  degradations from implicit conversions: because moving large data structures in and out of the JVM or
  between devices (e.g. a CPU and a GPU) can be costly, requiring explicit construction of data on the device
  makes it easier to debug performance problems.
- Scalars are 0-dimensional and are considered immutable for many operations.
- Some functions require that an axis or axes be specified. Axes are identified by indices starting at 0. For
  matrices, rows are along axis 0 and columns are along axis 1. In some cases, functions can take multiple
  axes as a vector (e.g. [0 1] specifies rows and columns of a matrix).
- Shapes are vectors of integers. The shape of a scalar is `[]` or `nil`. The shape of a vector is a
  1-dimensional Clojure vector (e.g. `[3]` is the shape of a 3-element vector). `[7 3 2 6 9 11]` is the
  shape of a 6-dimensional tensor.
- The word "vector" sometimes applies to a `clojure.lang.PersistentVector` and sometimes to a 1-dimensional
  `Tensure`. The type should be clear from context.
- As discussed above, functions suffixed with `!` mutate an argument. Generally it is the first argument that
  is mutated.
- Functions that mutate an argument often also return the mutated input argument. For instance, `(m/add! a b)`
  evaluates to `a`.

Below are summaries and examples of tensor functions. Consult the [API docs](https://cguenthner.github.io/tensure/docs/index.html) for details.

### Broadcasting

nd4j and `core.matrix` both support "broadcasting", as do other popular tensor libraries, such as
[numpy](https://github.com/numpy/numpy). "Broadcasting" is duplication of a tensor along some axis(es) to
match a particular shape. The rules for which shapes can be broadcast to which shapes differ between
[`core.matrix`](https://github.com/mikera/core.matrix/wiki/Broadcasting) and
[`numpy`](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html). Tensure implements its own
broadcasting functionality with rules that are similar but not identical to those used by numpy. Broadcasting
is most easily understood by looking at examples, so you may wish to skip the more formal description and just
look at the examples below. The algorithm for determining if shape `a` can be broadcast to shape `b` is as
follows:

1. If `a` is of higher dimensionality than `b`, then `a` cannot be broadcast to `b`.
2. Otherwise, `a` and `b` are aligned such that the last (_i_-th) dimension of `a` is matched with the last
   (_j_-th) dimension of `b`, the (_i_-1)th dimension of `a` is matched with the (_j_-1)-th dimension of `b`,
   etc.
3. `a` can be broadcast to `b` if the size of every dimension of `a` is equal to 1 or to the size of the
   aligned dimension of `b`.
4. If `a` cannot be broadcast to `b` with the present alignment, then `a` is shifted to the left relative to
   `b` such that the (_i_-th) dimension of `a` is matched with the (_j_-1)th dimension of `b`, the (_i_-1)th
   dimention of `a` is matched with the (_j_-2)th dimension of `b`, etc. The test in step 3 is repeated.
5. `a` is shifted left relative to `b` until a matching alignment is found, or until the 0th dimension of `a`
   is unmatched (i.e. until `a` has been shifted out of the range of `b`), in which case `a` cannot be
   broadcast to `b`.

The broadcasting operation itself is performed as follows:
1. Maintaining the alignment from above, shape `a` is padded with 1's at the beginning and end to match the
   dimensionality of `b`.
2. A tensor with shape `a` is reshaped to the padded shape found in step 1. (That is, some leading and
   trailing dimensions of size 1 may be added.)
3. For all dimensions where the broadcasted ("source") tensor has size 1 and the target shape has size > 1,
   the source tensor is repeated to match the size of the target shape.

Examples:

```
;; A 3-element vector can be broadcast to [2 3], because the shapes align as follows:
;; [2 3]
;;   [3]
(m/broadcast (m/array [1 2 3]) [2 3])
;; => #Tensure
;;    [[    1.0000,    2.0000,    3.0000],
;;     [    1.0000,    2.0000,    3.0000]]

;; A 1x3 matrix can also be broadcast to [2 3]:
;; [2 3]
;; [1 3]
(m/broadcast (m/array [[1 2 3]]) [2 3])
;; => #Tensure
;;    [[    1.0000,    2.0000,    3.0000],
;;     [    1.0000,    2.0000,    3.0000]]

;; A 2-element vector can be broadcast to [2 3], because the shapes align as follows:
;; [2 3]
;; [2]
(m/broadcast (m/array [1 2]) [2 3])
;; => #Tensure
;;    [[    1.0000,    1.0000,    1.0000],
;;     [    2.0000,    2.0000,    2.0000]]

;; A matrix with shape [1 2], cannot be broadcast to shape [2 3], because the shapes do not align:
;; X [2 3]
;; [1 2]X  <- This doesn't work
(m/broadcast (m/array [[1 2]]) [2 3])
;; => Exception

;; A 2-element vector can be broadcast to shape [2 2]. The vector is repeated along the leading axis, because
;; the vector's axis is paired with the trailing dimension:
;; [2 2]
;;   [2]
(m/broadcast (m/array [1 2]) [2 2])
;; => #Tensure
;;    [[    1.0000,    2.0000],
;;     [    1.0000,    2.0000]

;; In the example immediately above, if you wanted the vector to be repeated along the trailing axis (i.e. be
;; columns rather than rows), you would have to explicitly reshape it to make the dimensions match as desired:
;; [2 2]
;; [2 1]
(m/broadcast (m/reshape (m/array [1 2])
                        [2 1])
             [2 2])
;; => #Tensure
;;    [[    1.0000,    1.0000],
;;     [    2.0000,    2.0000]]
```

`broadcast` takes a source tensor and a target shape. Another function, `broadcast-like`, takes a source
tensor and a target tensor, and broadcasts the former to the shape of the latter. It's equivalent to
`(m/broadcast source-tensor (m/shape target-tensor))`.

Broadcasting can be done explicitly using `broadcast` or `broadcast-like` as demonstrated above. It is also
done implicitly by all elementwise arithmetic operations. As an example, consider elementwise addition using
`add`:

```
;; If adding a 3-element vector to a 2x3 matrix, the vector is implicitly broadcast to 2x3 before the
;; addition is performed.
(m/add (m/array [[1 2 3]
                 [4 5 6]])
       (m/array [10 20 30]))
;; => #Tensure
;;    [[   11.0000,   22.0000,   33.0000],
;;     [   14.0000,   25.0000,   36.0000]]

(m/add (m/array [[1 2 3]
                 [4 5 6]])
       (m/array [10 20]))
;; => #Tensure
;;    [[   11.0000,   12.0000,   13.0000],
;;     [   24.0000,   25.0000,   26.0000]]
```

Mutating arithmetic operations also perform implicit broadcasting, but the largest of the operands must be
first. Otherwise, the result of the operation would not fit in the desired location.

```
(def a (m/array [[1 2 3]
                 [4 5 6]]))
(def b (m/array [1 2 3]))

(m/add! b a)
;; => Exception
;; The result would be shape [2 3], but the first argument is of shape [3].

(m/add! a b)
a ; Evaluate a
;; => #Tensure
;;    [[    2.0000,    4.0000,    6.0000],
;;     [    5.0000,    7.0000,    9.0000]]
```

Operations on tensors and scalars can be considered a special case of broadcasting:

```
(m/add (m/array [[1 2 3]
                 [4 5 6]])
       (m/array 7))
;; => #Tensure
;;    [[    8.0000,    9.0000,   10.0000],
;;     [   11.0000,   12.0000,   13.0000]]

```

Broadcasting in Tensure differs from broadcasting in numpy and core.matrix in a few ways.

- In numpy _two_ tensors can be broadcast to create a result that is a larger shape than either operand. For
  instance, in numpy operands with the following shapes can be implicilty broadcast to produce a result of
  the indicated shape:

  ```
  Operand A Shape: [8 1 6 1]
  Operand B Shape:   [7 1 5]
  Result:          [8 7 6 5] <- This works in numpy but not in Tensure.
  ```

  In contrast, in Tensure one tensor must be strictly smaller than the other, and the result of an
  elementwise operation will always have the same shape as the argument with more elements. Numpy
  effectively performs broadcasting at the level of an axis, while Tensure performs broadcasting at the
  level of an entire tensor. In this sense, broadcasting in Tensure is more restrictive than broadcasting in
  numpy.
- Both numpy and core.matrix require that the trailing dimensions of the target shape be matched with a
  dimension of the source shape, while Tensure will append dimensions of size 1 to the source shape in order
  to match the target shape. For instance, as shown above, in Tensure a 2-element vector (shape [2]) can be
  broadcast to shape [2 3]: the vector is repeated as columns to produce a matrix of shape [2 3]. Neither
  numpy nor core.matrix would support this operation directly. Numpy would require that the vector be reshaped
  to [2 1] before being broadcast to [2 3]. It could be achieved in core.matrix only by permuting the axes:
  `(m/transpose (m/broadcast (m/array [1 2]) [3 2]))` (i.e. the 2-element vector is broadcast to shape [3 2],
  and then the axes of the results are swapped). In this sense, broadcasting in Tensure is more liberal than
  broadcasting in numpy and core.matrix.
- core.matrix will not broadcast a dimension of size 1 to some greater size. The shape of the source tensor
  must match exactly the trailing dimensions of the target tensor. For instance, core.matrix will not directly
  broadcast a tensor of shape [1 3] to shape [2 3], though it can broadcast a vector of shape [3] to shape [2
  3].

Implicit broadcasting makes it simpler to specify many operations, but it can also make it more difficult to
catch bugs. If there were no implicit broadcasting, any shape mismatch in the arguments to an elementwise
operation would generate an exception. With implicit broadcasting, some shape mismatches lead to silent
broadcasting. While writing code using Tensure, it's important to keep track of the expected shapes of
intermediate results to avoid bugs produced by implicit broadcasting when it is not intended. In general, when
performing complex tensor manipulations, assertions that intermediate results are of the desired shape can
catch many problems.

### Arithmetic

| Copying      | Mutating | Operation          |
| ------------ | -------- | ------------------ |
| `add`        | `add!`   | +                  |
| `sub`        | `sub!`   | –                  |
| `mul`        | `mul!`   | *                  |
| `div`        | `div!`   | /                  |
| `negate`     |          | unary –            |
| `pow`        | `pow!`   | ^/**               |
| `abs`        | `abs!`   |                    |
| `log`        | `log!`   | base e log         |
| `log10`      | `log10!` | base 10 log        |
| `sqrt`       | `sqrt!`  | √                  |
| `min`        | `min`    | Elementwise minimum|
| `max`        | `max`    | Elementwise maximum|
|                                              |
| **Trigonometric**                            |
| `cos`        | `cos!`   | cosine             |
| `sin`        | `sin!`   | sine               |
| `tan`        | `tan!`   | tangent            |
| `acos`       | `acos!`  | arccosine          |
| `asin`       | `asin!`  | arcsine            |
| `atan`       | `atan!`  | arctangent         |
|                                              |
| **Hyperbolic**                               |
| `cosh`       | `cosh!`  | hyperbolic cosine  |
| `tanh`       | `tanh!`  | hyperbolic tangent |
| `sinh`       | `sinh!`  | hyperbolic sine    |

### Comparison

Elementwise comparison operators only receive two arguments. Boolean values in Tensure are represented as
floats: 1.0 for `true` and 0.0 for `false`.

| Copying      | Mutating | Operation          |
| ------------ | -------- | ------------------ |
| `eq`         | `eq!`    | =                  |
| `ne`         | `ne!`    | !=                 |
| `gt`         | `gt!`    | >                  |
| `lt`         | `lt!`    | <                  |
| `le`         | `le!`    | <=                 |
| `ge`         | `ge!`    | >=                 |

`equals` is semantically equivalent to the standard Clojure `=` when applied to tensors.

### Linear algebra

| Copying        | Operation      |
| -------------- | -------------- |
| `mmul`         | _scalars_: product <br> _vectors_: inner product <br> _matrices_: matrix product <br> _tensors_: tensor dot product |
| `outer-product`| outer product  |

### Reducing

Reducing functions with names prefixed with an _e_ take a single tensor and return a scalar aggregate value
over all elements in that tensor:

| Over all elements |                    |
| ----------------- | ------------------ |
| `emax`            | maxmium            |
| `emin`            | minimum            |
| `esum`            | sum                |
| `emean`           | arithmetic mean    |
| `estdev`          | standard deviation |

Reducing functions with names suffixed with _-along_ aggregate values along an axes or axes. They all have
two arities like:

```
(sum along tensor axes)
(sum-along tesnor axes collapse)
```

where `axes` is a single axis or an array of `axes`, and `collapse` is a boolean (default `true`) indicating
whether or not to remove the reduced dimensions.

| Over an axis/axes |                    |
| ----------------- | ------------------ |
| `sum-along`       | sum                |
| `max-along`       | maximum            |
| `min-along`       | minimum            |

Examples:

```
(def a (m/array [[1 2] [3 4]]))
(def c (m/sum-along a 0))  ; => #Tensure [[    4.0000,    6.0000]]
(m/shape c)                ; => [2]

(def d (m/sum-along a 1))  ; => #Tensure [[    3.0000,    7.0000]]
(m/shape d)                ; => [2]

(def e (m/sum-along a [0 1])) ; => #Tensure 10.0000
(m/shape e)                   ; => nil

(def f (m/sum-along a 0 false)) ; => #Tensure [[    4.0000,    6.0000]]
(m/shape f)                     ; => [1 2]

(def g (m/sum-along a 1 false)) ; => #Tensure [3.0000,
                                               7.0000]
(m/shape g)                     ; => [2 1]

(def h (m/sum-along a [0 1] false)) ; => #Tensure 10.0000
(m/shape h)                         ; => [1 1]
```

 `argmax-along` and `argmin-along` accept a tensor and an axis/axes and redurn the indices of the minimum and
 maximum. If multiple axes are provided, then the index returned is the index within the reduced axes, in row
 major order.

| Over an axis/axes |                    |
| ----------------- | ------------------ |
| `argmax-along`    | argmax             |
| `argmin-along`    | argmin             |

Examples:
```
(def a (m/array [[1 3 2]
                 [0 1 3]
                 [0 3 4]]))
(def b (m/argmax-along a 0))  ; => #Tensure [         0,         0,    2.0000]
(m/shape b)  ; => [3]

(m/argmax-along a 1)  ; => #Tensure [    1.0000,    2.0000,    2.0000]

(def c (m/argmax-along a [0 1]))  ; => #Tensure 8.0000
(m/shape c)  ; => nil

(m/argmin-along a 0)  ; => #Tensure [    1.0000,    1.0000,         0]
(m/argmin-along a 1)  ; => #Tensure [         0,         0,         0]
(m/argmin-along a [0 1])  ; => #Tensure 3.0000
```

### Selection
**`submatrix`** returns a view over a subregion of a tensor:
```
(def a (m/array [[1 3 2]
                 [0 1 3]
                 [0 3 4]]))

; Select 2 rows (along dimension 0) starting at index 1, and keep all slices through other dimensions.
; The arguments are like (m/submatrix tensor dimension-index [start-index length]).
(def b (m/submatrix a 0 [1 2]))
;; => #Tensure [[         0,    1.0000,    3.0000],
;;              [         0,    3.0000,    4.0000]]
(m/shape b)  ; => [2 3]
(m/same-data? a b)  ; => true

; Select 1 column (along dimension 1) starting at index 2
(def c (m/submatrix a 1 [2 1]))
;; => #Tensure [2.0000,
;;              3.0000,
;;              4.0000]
(m/shape c)  ; => [3 1]

; You can alternatively provide a seq of tuples like [start-index length] for multiple axes.
; Select 1 row and 2 columns starting at [2, 0].
(m/submatrix a [[2 1] [0 2]])
;; => #Tensure [[         0,    3.0000]]

; For matrices, there is a special arity like:
; (m/submatrix row-start row-count col-start col-count)
; Starting at row 0, select 3 rows; starting at column 2, select 1 column
(m/submatrix 0 3 2 1)
;; => #Tensure [2.0000,
;;              3.0000,
;;              4.0000]
```

**`select-range`** accepts a tensor and a number of additional arguments equal to the tensor's
dimensionality. Each argument can be:

  - _a number_ - to select a slice through that dimension, eliminating the dimension
  - `[start stop]` - to select a continuous range of slices along that dimension (inclusive of `start`,
    exclusive of `stop`)
  - `[start stop step]` - to select every `step`-th slice in [`start`, `stop`)
  - _a keyword_ - one of the following: `:first`, `:last`, `:all`, `:butlast`, `:rest`

The ranges accepted by `select-range` are semantically equivalent to the arguments accepted by
`clojure.core/range`.

```
(def a (m/array [[1 2 3 4]
                 [5 6 7 8]
                 [9 10 11 12]
                 [13 14 15 16]]))
(def b (m/select-range a [1 4] [1 3]))
;; => #Tensure [[    6.0000,    7.0000],
;;              [   10.0000,   11.0000],
;;              [   14.0000,   15.0000]]
(m/same-data? a b)  ; => true

(def c (m/select-range a [0 5 2] [0 3 2]))
;; => #Tensure [[    1.0000,    3.0000],
;;              [    9.0000,   11.0000]]
(m/same-data? a c)  ; => true

;; Different types of selections can be used for different dimensions.
(def d (m/select-range a 1 [2 4]))  ; => #Tensure [[    7.0000,    8.0000]]
(m/shape d)  ; => 2

;; Note that a single-index selection eliminates a dimension, while a range selection includes all dimensions.
(def e (m/select-range a 2 3)) ; => #Tensure 12.0000
(m/shape e)  ; => nil

(def f (m/select-range a [2 2] [3 3]))  ; => #Tensure 12.0000
(m/shape f)  ; => [1 1]

(def g (m/select-range a :butlast :first))  ; => #Tensure [[    1.0000,    5.0000,    9.0000]]
(m/shape g)  ; => [3]
```

**`select-axis-range`** is like `select-range`, except that it takes a selection through only a single
dimension, taking arguments like: `(m/select-axis-range tensor axis selection)`. Continuing the example above,

```
(m/select-axis-range a 0 2)  ; => #Tensure [[    9.0000,   10.0000,   11.0000,   12.0000]]
(m/select-axis-range a 1 [1 3])  ; => #Tensure [[    2.0000,    3.0000],
                                                [    6.0000,    7.0000],
                                                [   10.0000,   11.0000],
                                                [   14.0000,   15.0000]]
(m/select-axis-range a 0 :first)  ; => #Tensure [[    1.0000,    2.0000,    3.0000,    4.0000]]
```

There are analagous functions, `set-range!` and `set-axis-range!`, that take an additional argument, a tensor
of the same size as the selection, and set the selected elements to the corresponding values in that
tensor. See the information below on [setting operations](#setting).

### Splitting and joining

There are a few functions that return sequences of tensors:

| Function          | Description                                                                          |
| ----------------- | ------------------------------------------------------------------------------------ |
| `slices`          | `(m/slices tensor axis-index)` returns a seq of slices along some dimension. Each slice has one fewer dimensions than the input tensor. |
| `rows`            | `(m/rows tensor)` returns a seq of slices through the first (index 0) dimension (rows if the tensor is a matrix) |
| `columns`         | `(m/columns tensor)` returns a seq of slices through the second (index 1) dimension (columns if the tensor is a matrix.) |
| `partition-along` | `(m/partition-along tensor axis-index partition-size step-size)` splits the tensor into chunks of `partition-size` spaced by `step-size` along some axis. Analagous to `clojure.core/partition-all` but for tensors and an arbitrary axis. |

Examples:
```
(def a (m/array [[1 2 3] [4 5 6] [7 8 9]]))
(m/slices a 0)
;; => (#Tensure [[    1.0000,    2.0000,    3.0000]]
;;     #Tensure [[    4.0000,    5.0000,    6.0000]]
;;     #Tensure [[    7.0000,    8.0000,    9.0000]])

(def b (second (m/rows a)))  ; => #Tensure [[    4.0000,    5.0000,    6.0000]]
(m/shape b)  ; => [3]

(def c (m/partition-along (m/array [0 1 2 3 4 5 6 7 8]) 0 3 2))
;; => (#Tensure [[         0,    1.0000,    2.0000]]
;;     #Tensure [[    2.0000,    3.0000,    4.0000]]
;;     #Tensure [[    4.0000,    5.0000,    6.0000]]
;;     #Tensure [[    6.0000,    7.0000,    8.0000]]
;;     #Tensure 8.0000)
(map m/shape c)  ; => ([3] [3] [3] [3] [1])
```

The above functions take a single tensor and return multiple tensors. **`join-along`** and **`stack`** take
multiple tensors and return a single tensor produced by concatenating the input tensors along some
dimension. They differ in whether or not they add the concatenated dimension: **`join-along`** concatenates
along an existing dimension, while **`stack`** concatenates along a new dimension:

```
(def a (m/join-along 0 (m/array [1 2]) (m/array [3 4]) (m/array [5 6])))
;; => #Tensure [[    1.0000,    2.0000,    3.0000,    4.0000,    5.0000,    6.0000]]
(m/shape a)  ; => [6]

(def b (m/stack 0 (m/array [1 2]) (m/array [3 4]) (m/array [5 6])))
;; => #Tensure [[    1.0000,    2.0000],
;;              [    3.0000,    4.0000],
;;              [    5.0000,    6.0000]]
(m/shape b)  ; => [3 2]

(def c (m/stack 1 (m/array [1 2]) (m/array [3 4]) (m/array [5 6])))
;; => #Tensure [[    1.0000,    3.0000,    5.0000],
;;              [    2.0000,    4.0000,    6.0000]]
(m/shape c)  ; => [2 3]
```

### Reshaping and permuting
**`reshape`** transforms a tensor into a provided shape, which must have the same number of elements as the
tensor.

```
(def a (m/reshape (m/array [1 2 3 4]) [2 2]))
;; => #Tensure [[    1.0000,    2.0000],
;;              [    3.0000,    4.0000]]
(m/shape a)  ; => [2 2]
```

**`permute`** and its mutating analog **`permute!`** reorder dimensions. The second argument is a seq of
indices for the input tensor's current axes, reordered as desired. For example, with `[2 1 0]` as the second
argument, axis 2 of the input tensor will be the 0-th dimension of the permuted tensor, axis 1 will be the
1-st dimension, and axis 0 will be the 2-nd dimension:

```
(def a (m/array [[[1 2] [3 4]]
                 [[5 6] [7 8]]
                 [[9 10] [11 12]]]))
(def b (m/permute a [2 1 0]))
;; => #Tensure [[[    1.0000,    5.0000,    9.0000],
;;               [    3.0000,    7.0000,   11.0000]],
;;              [[    2.0000,    6.0000,   10.0000],
;;               [    4.0000,    8.0000,   12.0000]]]
(m/same-data? a b)  ; => true
```

**`add-dimension`** inserts a dimension of size 1 at some axis index (defaults to 0).
```
(def a (m/array [1 2 3]))
(def b (m/add-dimension a))  ; => #Tensure [[    1.0000,    2.0000,    3.0000]]
(m/shape b)  ; => [1 3]

(def c (m/add-dimension a 1))  ; => #Tensure [[    1.0000,    2.0000,    3.0000]]
(m/shape c)  ; => [3 1]
```

**`transpose`** returns that standard tensor transpose.

### Mapping

Tensure includes two functions, **`emap`** and **`emap-indexed`**, for constructing new tensors by mapping
over the elements of an input tensor (or multiple input tensors), similiar to how `clojure.core/map` and
`clojure.core/map-indexed` map over Clojure seqs:

```
(def a (m/array [[1 2]
                 [3 4]]))
(def b (m/array [[10 20]
                 [30 40]]))
(m/emap (fn [a b]
         (+ a b))
       a b)
;; => #Tensure [[   11.0000,   22.0000],
;;              [   33.0000,   44.0000]]

(m/emap-indexed (fn [indices a b]
                  (println indices a b)
                  (+ a b (* 100 (apply + indices))))
                a b)
;; => #Tensure [[   11.0000,  122.0000],
;;              [  133.0000,  244.0000]]
;; Prints:
;; [0 0] 1.0 10.0
;; [0 1] 2.0 20.0
;; [1 0] 3.0 30.0
;; [1 1] 4.0 40.0
```

Note that `emap` and `emap-indexed` require moving data into and out of the JVM, which is slow. They also are
not effecitvely parallelized. They therefore have very poor performance, and, if possible, it would be better
to use other Tensure functions:

```
(def c (m/sample-uniform [100 100]))
(def d (m/sample-uniform [100 100]))

(time (m/emap + c d))
;; "Elapsed time: 10.153322 msecs"

(time (m/add c d))
;; "Elapsed time: 0.464575 msecs"

```

### Setting
| Function          | Description                                                                          |
| ----------------- | ------------------------------------------------------------------------------------ |
| `set-range!`      | Analagous to `select-range` but receives an additional argument, a source tensor the same size as the selection, and sets the elements in the selected range to the corresponding values in the source tensor |
| `set-axis-range!` | Analagous to `select-axis-range` |
| `assign!`         | `(assign! a b)` sets all elements of `a` to the corresponding values in `b`. `a` and `b` must be the same size, though `a` can be a view over a larger tensor (e.g. produced with `submatrix` or `select-range`) |
| `fill!`           | `(fill! a n)` sets every element in tensor `a` to number `n`. |
| `mset!`           | `(mset! a i i+1 i+2 ... n)` sets the element at position [i, i+1, i+2 ...] to value `n`. |
| `mset`            | Like `mset!` but copies the input data |

Examples:
```
(def a (m/array [[1 2 3 4]
                 [5 6 7 8]
                 [9 10 11 12]
                 [13 14 15 16]]))
(m/fill! a 7)
a ; Evaluate a
;; => #Tensure [[    7.0000,    7.0000,    7.0000,    7.0000],
;;              [    7.0000,    7.0000,    7.0000,    7.0000],
;;              [    7.0000,    7.0000,    7.0000,    7.0000],
;;              [    7.0000,    7.0000,    7.0000,    7.0000]]

(m/assign! a (m/array [[1 2 3 4]
                       [5 6 7 8]
                       [9 10 11 12]
                       [13 14 15 16]]))
;; => #Tensure [[    1.0000,    2.0000,    3.0000,    4.0000],
;;              [    5.0000,    6.0000,    7.0000,    8.0000],
;;              [    9.0000,   10.0000,   11.0000,   12.0000],
;;              [   13.0000,   14.0000,   15.0000,   16.0000]]

(m/set-range! a [0 4 2] [0 4 2] (m/zeros [2 2]))
;; => #Tensure [[         0,    2.0000,         0,    4.0000],
;;              [    5.0000,    6.0000,    7.0000,    8.0000],
;;              [         0,   10.0000,         0,   12.0000],
;;              [   13.0000,   14.0000,   15.0000,   16.0000]]

(m/mset! a 0 0 1)
(m/mset! a 0 2 3)
(m/mset! a 2 0 9)
(m/mset! a 2 2 11)
a ; Evaluate a
;; => #Tensure [[    1.0000,    2.0000,    3.0000,    4.0000],
;;              [    5.0000,    6.0000,    7.0000,    8.0000],
;;              [    9.0000,   10.0000,   11.0000,   12.0000],
;;              [   13.0000,   14.0000,   15.0000,   16.0000]]

(m/assign! (m/submatrix a [[1 2] [1 2]]) (m/zeros [2 2]))
;; => #Tensure [[    1.0000,    2.0000,    3.0000,    4.0000],
;;              [    5.0000,         0,         0,    8.0000],
;;              [    9.0000,         0,         0,   12.0000],
;;              [   13.0000,   14.0000,   15.0000,   16.0000]]

(m/set-axis-range! a 1 [1 3] (m/filled [4 2] 7))
;; => #Tensure [[    1.0000,    7.0000,    7.0000,    4.0000],
;;              [    5.0000,    7.0000,    7.0000,    8.0000],
;;              [    9.0000,    7.0000,    7.0000,   12.0000],
;;              [   13.0000,    7.0000,    7.0000,   16.0000]]
```

### Misc
**`shift`** shifts the elements in a tensor along some dimension(s) by some number of elements:

```
(shift tensor dimension shift-amount)
(shift tensor shift-amounts)
```

where `shift-amount` is an integer and `shift-amounts` is a seq of integers equal in length to the
dimensionality of `tensor`. Positive shift amounts move the elements up/left, and negative shift amounts move
the elements down/right; if this is confusing, then think of the shift amounts as moving a "camera" looking at the tensor--moving the camera down/right (a positive shift of the camera) makes the elements appear to move up/left in the frame of the camera. The elements left vacant after the shift are filled with 0.

```
(def a (m/array [[1 2 3 4]
                 [5 6 7 8]
                 [9 10 11 12]
                 [13 14 15 16]]))

(m/shift a 0 -1)
;; => #Tensure [[         0,         0,         0,         0],
;;              [    1.0000,    2.0000,    3.0000,    4.0000],
;;              [    5.0000,    6.0000,    7.0000,    8.0000],
;;              [    9.0000,   10.0000,   11.0000,   12.0000]]

(m/shift a 0 1)
;; => #Tensure [[    5.0000,    6.0000,    7.0000,    8.0000],
;;              [    9.0000,   10.0000,   11.0000,   12.0000],
;;              [   13.0000,   14.0000,   15.0000,   16.0000],
;;              [         0,         0,         0,         0]]

; Shift 1 element up on axis 0 and 2 elements right on axis 1.
(m/shift a 0 [1 -2])
;; => #Tensure [[         0,         0,    5.0000,    6.0000],
;;              [         0,         0,    9.0000,   10.0000],
;;              [         0,         0,   13.0000,   14.0000],
;;              [         0,         0,         0,         0]]
```

## Contributing

If you find a bug or would like a feature, then please open a GitHub issue. Even better, please submit a pull
request. There is quite a bit of low-hanging fruit:

- Add support for double-precision arrays
- Add support for the CUDA backend
- Add additional functions and operations
- Expand test coverage
- Benchmark and investgiate possible performance improvements
- Improve compatibility with `core.matrix`
- Implement `core.matrix` interfaces

## License

Copyright © 2019 Casey Guenthner

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
