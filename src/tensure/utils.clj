(ns tensure.utils)

(defn throw-str
  "Throws an `Exception` with a message produced by `str`ing together `args`."
  [& args]
  (let [; Printing giant Exception messages freezes Cider, so we truncate them after a certain length.
        max-throw-str-len 10000
        ; Ellipsis is three characters.
        max-msg-len (- max-throw-str-len 3)
        s (apply str args)
        message (if (<= (count s) max-msg-len)
                  s
                  (str (subs s 0 max-msg-len) "..."))]
    (throw (Exception. message))))

(defn zip
  "Zips two collections together. E.g. (zip [1 2 3] [4 5 6]) returns ([1 4] [2 5] [3 6])."
  [& colls]
  (if (zero? (count colls))
    []
    (apply map vector colls)))

(defn unchunk
  "Returns a lazy-seq that does not use chunking."
  [s]
  (when (seq s)
    (lazy-seq
      (cons (first s)
            (unchunk (next s))))))
