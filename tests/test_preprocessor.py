def test_fit_transform():
  df = pd.DataFrame({

  })
  prep = TitanicPreprocessor()
  X, y = prep.fit_transform(df)
  assert X.shape[0] == len(y)