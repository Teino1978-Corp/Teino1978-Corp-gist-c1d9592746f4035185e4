 def update(td: TaggingDistribution, ad: AlignmentDistribution) = {
    /*
    * Setup: Create an empty set of records
    */
    var TPrime = List[Record]()
    for (i <- 0 until _T.size) {
      val r = new Record(i, _schema)
      r.setThetaRewrite(_T(i).thetaRewrite)
      TPrime = TPrime :+ r
    }

    val c = new Counter[String]()
    for (tweet <- 0 until td.tweets.size) {
      updateForTweet(tweet, TPrime, td, ad, c)
    }
    // logger.info("Counts")
    // logger.info(c.toString())
    
    /*
    * Final:
    * Normalize per tweet
    */
    for (record <- TPrime) {
      for ((fieldName, multinomial) <- record.fields) {
        multinomial.counter.logNormalize
      }
    }

    _T = TPrime
    // Make sure each record-field is a probabiility distribution
    checkOK
  }



  def updateForTweet(tweet: Int, tPrime: List[Record], td: TaggingDistribution, ad: AlignmentDistribution, c:Counter[String]) = {
    /*
     *  Initialization
     */

    val stateSpace = td.crf.crf.getStateSpace()
    val fb: edu.umass.nlp.ml.sequence.ForwardBackwards = new edu.umass.nlp.ml.sequence.ForwardBackwards(stateSpace)

    // NumRecords x NumFields x [Field Value -> Potentials]
    // 100 x 3 x 50 x 30 x 16
    //var recordPotentials = List[List[Map[List[String],Array[Array[Double]]]]]()
    val potentials = td.crfPotentialsCopy(tweet)
    // val boosts = thetaRewrite(tweet)
    // val allRecordBoosts: Map[Record,Seq[Boost]] = boosts.groupBy(_.record)
    for ((record,recordIndex) <- tPrime.zipWithIndex) {
      val boosts = record.thetaRewrite(tweet)
      val alignmentBelief = ad.Z(tweet)(recordIndex) // q(z_i = k)
      if (boosts.size > 0) {
          var foo = Map[String, Double]()
          
          
          for (boost <- boosts) {
              // counter.logInc does a logAdd
              // eg: newValue = log ( exp(currentValue) + exp(argument) )
              tPrime(recordIndex).fields(boost.recordField).counter.logInc(boost.fieldValue,math.log(alignmentBelief * boost.similarityScore))
              // tPrime(recordIndex).fields(boost.recordField).counter.inc(boost.fieldValue,alignmentBelief * boost.similarityScore)
          }
          
          
          for ((f,vals) <- record.fields) {
            for (v <- vals.counter.counts.keys) {
              updatePotentials(potentials,td,boosts,f,v,scale=1.0)
              fb.setInput(potentials)
              tPrime(recordIndex).fields(f).counter.dec(v,math.log(alignmentBelief) + fb.getLogZ)
              updatePotentials(potentials,td,boosts,f,v,scale= -1.0)
            }
          }
          
          
      }
    }