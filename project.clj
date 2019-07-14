(defproject tensure "0.1.0"
  :description "A high-level tensor math library for Clojure."
  :url "https://github.com/cguenthner/tensure"
  :license {:name "MIT License"
            :url "https://opensource.org/licenses/MIT"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [org.nd4j/nd4j-native-platform "1.0.0-beta3"]
                 [org.nd4j/nd4j-api "1.0.0-beta3"]]
  :repl-options {:init-ns tensure.core}
  :codox {:namespaces [tensure.core]})
