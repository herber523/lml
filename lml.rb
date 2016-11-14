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

  class NB
    def initialize(data)
      @data_rows = load_data(data)
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

    private

    def load_data(data)
      data_rows = []
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
        data_rows << { mean: mean, variance: variance, type: type }
      end
      data_rows
    end

    def pdf(n, mean, variance)
      (1 / Math.sqrt(3.1415 * variance)) * Math.exp(-((n - mean)**2) / variance)
    end
  end

  class APRIORI
    def initialize(data)
      @data = data.map { |n| n[:data] }
    end

    def apriori(min_support)
      data = @data
      c1 = find_item
      l1, support_data = find_support(data, c1, min_support)
      l = [l1]
      k = 0
      while l[k].count > 0
        ck = apriori_gen(l[k], k)
        lk, supk = find_support(data, ck, min_support)
        support_data = support_data.merge(supk)
        l << lk
        k += 1
      end
      [l, support_data]
    end

    private

    def find_item
      set = []
      @data.each do |transction|
        transction.each do |item|
          set << item unless set.include?(item)
        end
      end
      set.sort
    end

    def find_support(d, ck, min_support)
      sscnt = {}
      d.each do |tid|
        ck.each do |can|
          can = [can] unless can.is_a?(Array)
          if (can - tid).empty?
            sscnt[can] = 0 if sscnt[can].nil?
            sscnt[can] += 1
          end
        end
      end
      items_num = d.count.to_f
      support_data = {}
      retlist = []
      sscnt.keys.each do |key|
        support = sscnt[key] / items_num
        retlist << key if support >= min_support
        support_data[key] = support
      end
      [retlist, support_data]
    end

    def apriori_gen(l1, k)
      d = []
      lc = l1.count
      (0..lc - 1).each do |i|
        (i + 1..lc - 1).each do |j|
          if l1[i].is_a?(Array)
            d << (l1[i] | l1[j]).sort if l1[i].first(k) == l1[j].first(k)
          else
            d << ([l1[i]] | [l1[j]]).sort
          end
        end
      end
      d.sort
    end
  end
end
