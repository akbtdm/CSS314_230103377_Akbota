define i32 @_ZN8__main__18worker_strided_lcgB2v5B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedExxy(i8** noalias nocapture writeonly %retptr, { i8*, i32, i8*, i8*, i32 }** noalias nocapture readnone %excinfo, i8* nocapture readnone %arg.buf.0, i8* nocapture readnone %arg.buf.1, i64 %arg.buf.2, i64 %arg.buf.3, i64* nocapture %arg.buf.4, i64 %arg.buf.5.0, i64 %arg.buf.6.0, i64 %arg.slot, i64 %arg.iters, i64 %arg.seed) local_unnamed_addr #0 {
B0.endif:
  %.1063.not = icmp slt i64 %arg.iters, 1
  br i1 %.1063.not, label %common.ret, label %B30.lr.ph

B30.lr.ph:                                        ; preds = %B0.endif
  br label %B30

B30:                                              ; preds = %B30.lr.ph, %B30
  %.48.04 = phi i64 [ 0, %B30.lr.ph ], [ %.119, %B30 ]
  %.119 = add nuw nsw i64 %.48.04, 1
  %sunkaddr = mul i64 %arg.slot, 8
  %0 = bitcast i64* %arg.buf.4 to i8*
  %sunkaddr6 = getelementptr i8, i8* %0, i64 %sunkaddr
  %1 = bitcast i8* %sunkaddr6 to i64*
  %.163 = load atomic i64, i64* %1 monotonic, align 8
  %.164 = mul i64 %.163, 6364136223846793005
  %.166 = xor i64 %.48.04, %arg.seed
  %.167 = add i64 %.164, %.166
  store atomic i64 %.167, i64* %1 monotonic, align 8
  %exitcond.not = icmp eq i64 %arg.iters, %.119
  br i1 %exitcond.not, label %common.ret, label %B30

common.ret:                                       ; preds = %B30, %B0.endif
  store i8* null, i8** %retptr, align 8
  ret i32 0
}
