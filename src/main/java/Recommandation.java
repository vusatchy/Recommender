import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.DataModelBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.CachingRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDPlusPlusFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class Recommandation {

    public static void main(String[] args) throws IOException, TasteException {
        DataModel dataModel = new FileDataModel(new File("D:\\Newton\\Recommender\\src\\main\\resources\\joke.txt"));
        UserSimilarity userSimilarity = new PearsonCorrelationSimilarity(dataModel);
        UserNeighborhood neighborhood =
                new NearestNUserNeighborhood(3, userSimilarity, dataModel);
        Recommender recommender = new GenericUserBasedRecommender(dataModel, neighborhood, userSimilarity);
        Recommender cachingRecommender = new CachingRecommender(recommender);
        List<RecommendedItem> recommendations =
                cachingRecommender.recommend(1, 1);
        for (RecommendedItem recommendedItem : recommendations) {
            System.out.println(recommendedItem.getItemID());
        }

        RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {
            public Recommender buildRecommender(DataModel dataModel) throws TasteException {
                UserSimilarity userSimilarity = new PearsonCorrelationSimilarity(dataModel);
                UserNeighborhood neighborhood =
                        new NearestNUserNeighborhood(3, userSimilarity, dataModel);
                Recommender recommender = new GenericUserBasedRecommender(dataModel, neighborhood, userSimilarity);
                Recommender cachingRecommender = new CachingRecommender(recommender);
                return recommender;
            }
        };
        RecommenderEvaluator recommenderEvaluator = new RMSRecommenderEvaluator();
        double eval = recommenderEvaluator.evaluate(recommenderBuilder, null, dataModel, 0.9, 1.0);
        System.out.println(eval);
        System.out.println("+-----------------------------------------------+");
        ItemSimilarity itemSimilarity = new PearsonCorrelationSimilarity(dataModel);
        recommender = new GenericItemBasedRecommender(dataModel, itemSimilarity);
        cachingRecommender = new CachingRecommender(recommender);
        recommendations = cachingRecommender.recommend(1, 1);
        for (RecommendedItem recommendedItem : recommendations) {
            System.out.println(recommendedItem.getItemID());
        }
        recommenderBuilder = new RecommenderBuilder() {
            public Recommender buildRecommender(DataModel dataModel) throws TasteException {
                ItemSimilarity itemSimilarity = new PearsonCorrelationSimilarity(dataModel);
                Recommender recommender = new GenericItemBasedRecommender(dataModel, itemSimilarity);
                Recommender cachingRecommender = new CachingRecommender(recommender);
                return cachingRecommender;
            }
        };
        eval = recommenderEvaluator.evaluate(recommenderBuilder, null, dataModel, 0.9, 1.0);
        System.out.println(eval);
        System.out.println("+----------------------------------------------+");
        recommender = new SVDRecommender(dataModel, new SVDPlusPlusFactorizer(dataModel, 100, 10000));
        cachingRecommender = new CachingRecommender(recommender);
        recommendations = cachingRecommender.recommend(1, 1);
        for (RecommendedItem recommendedItem : recommendations) {
            System.out.println(recommendedItem.getItemID());
        }
        recommenderBuilder = new RecommenderBuilder() {
            public Recommender buildRecommender(DataModel dataModel) throws TasteException {
                Recommender recommender = new SVDRecommender(dataModel, new SVDPlusPlusFactorizer(dataModel, 100, 10000));
                Recommender cachingRecommender = new CachingRecommender(recommender);
                return cachingRecommender;
            }
        };
        eval = recommenderEvaluator.evaluate(recommenderBuilder, null, dataModel, 0.9, 1.0);
        System.out.println(eval);

    }
}
