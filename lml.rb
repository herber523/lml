require 'distance_measures'
require 'ruby_native_statistics'
module LML
  class KNN
    def initialize(data)
      @data = data
    end

    def classify(input, k)
      knn = []
      @data.each do |d|
        # distance = d[:data].cosine_similarity(input[:data])
        distance = d[:data].tanimoto_coefficient(input[:data])
        knn << [distance, d[:type]]
      end
      knn.sort_by { |n| n[0] }.reverse.first(k)
    end
  end

  class NBC
    def initialize(data)
      @data_rows = []
      data_count = data[0][:data].count
      types = data.map { |n| n[:type] }.uniq
      types.each do |type|
        data_row = []
        mean = []
        variance = []
        (0..data_count - 1).each do |i|
          t = []
          data.each do |d|
            t << d[:data][i] if d[:type] == type
          end
          mean << t.inject(:+).to_f / t.count
          variance << 2 * t.var
        end
        @data_rows << { mean: mean, variance: variance, type: type }
      end
    end

    def classify(input)
      nbc = []
      @data_rows.each do |data_row|
        evidence = 1
        data_row[:mean].zip(data_row[:variance], input[:data]) do |mean, variance, input|
          evidence *= pdf(input, mean, variance)
        end
        nbc << { type: data_row[:type], num: evidence }
      end
      nbc
    end

    def pdf(n, mean, variance)
      (1 / Math.sqrt(3.1415 * variance)) * Math.exp(-((n - mean)**2) / variance)
    end
  end
end
