define i32 @_ZN8__main__25worker_private_accumulateB2v3B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedExxy(i8** noalias nocapture writeonly %retptr, { i8*, i32, i8*, i8*, i32 }** noalias nocapture readnone %excinfo, i8* nocapture readnone %arg.buf.0, i8* nocapture readnone %arg.buf.1, i64 %arg.buf.2, i64 %arg.buf.3, i64* nocapture %arg.buf.4, i64 %arg.buf.5.0, i64 %arg.buf.6.0, i64 %arg.slot, i64 %arg.iters, i64 %arg.seed) local_unnamed_addr #0 {
B0.endif:
  %.39 = icmp slt i64 %arg.slot, 0
  %.40 = select i1 %.39, i64 %arg.buf.5.0, i64 0
  %.41 = add i64 %.40, %arg.slot
  %.54 = getelementptr i64, i64* %arg.buf.4, i64 %.41
  %.55 = load i64, i64* %.54, align 8
  %0 = tail call i64 @llvm.smax.i64(i64 %arg.iters, i64 0)
  %.1403.not = icmp slt i64 %arg.iters, 1
  br i1 %.1403.not, label %common.ret, label %B38.preheader

B38.preheader:                                    ; preds = %B0.endif
  %min.iters.check = icmp ult i64 %0, 4
  br i1 %min.iters.check, label %B38.preheader14, label %vector.ph

vector.ph:                                        ; preds = %B38.preheader
  %n.vec = and i64 %0, 9223372036854775804
  %1 = insertelement <2 x i64> <i64 poison, i64 0>, i64 %.55, i64 0
  %broadcast.splatinsert = insertelement <2 x i64> poison, i64 %arg.seed, i64 0
  %broadcast.splat = shufflevector <2 x i64> %broadcast.splatinsert, <2 x i64> poison, <2 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %lsr.iv = phi i64 [ %lsr.iv.next, %vector.body ], [ %n.vec, %vector.ph ]
  %vec.phi = phi <2 x i64> [ %1, %vector.ph ], [ %4, %vector.body ]
  %vec.phi7 = phi <2 x i64> [ zeroinitializer, %vector.ph ], [ %5, %vector.body ]
  %vec.ind = phi <2 x i64> [ <i64 0, i64 1>, %vector.ph ], [ %vec.ind.next, %vector.body ]
  %step.add = add <2 x i64> %vec.ind, <i64 2, i64 2>
  %2 = xor <2 x i64> %vec.ind, %broadcast.splat
  %3 = xor <2 x i64> %step.add, %broadcast.splat
  %4 = add <2 x i64> %vec.phi, %2
  %5 = add <2 x i64> %vec.phi7, %3
  %vec.ind.next = add <2 x i64> %vec.ind, <i64 4, i64 4>
  %lsr.iv.next = add i64 %lsr.iv, -4
  %6 = icmp eq i64 %lsr.iv.next, 0
  br i1 %6, label %middle.block, label %vector.body, !llvm.loop !0

middle.block:                                     ; preds = %vector.body
  %bin.rdx = add <2 x i64> %5, %4
  %7 = call i64 @llvm.vector.reduce.add.v2i64(<2 x i64> %bin.rdx)
  %cmp.n = icmp eq i64 %0, %n.vec
  br i1 %cmp.n, label %common.ret, label %B38.preheader14

B38.preheader14:                                  ; preds = %B38.preheader, %middle.block
  %acc.2.06.ph = phi i64 [ %7, %middle.block ], [ %.55, %B38.preheader ]
  %.82.04.ph = phi i64 [ %n.vec, %middle.block ], [ 0, %B38.preheader ]
  br label %B38

B38:                                              ; preds = %B38.preheader14, %B38
  %acc.2.06 = phi i64 [ %.181, %B38 ], [ %acc.2.06.ph, %B38.preheader14 ]
  %.82.04 = phi i64 [ %.153, %B38 ], [ %.82.04.ph, %B38.preheader14 ]
  %.153 = add nuw nsw i64 %.82.04, 1
  %.179 = xor i64 %.82.04, %arg.seed
  %.181 = add i64 %acc.2.06, %.179
  %exitcond.not = icmp eq i64 %0, %.153
  br i1 %exitcond.not, label %common.ret, label %B38, !llvm.loop !2

common.ret:                                       ; preds = %B38, %middle.block, %B0.endif
  %acc.2.0.lcssa = phi i64 [ %.55, %B0.endif ], [ %7, %middle.block ], [ %.181, %B38 ]
  %sunkaddr = mul i64 %.41, 8
  %8 = bitcast i64* %arg.buf.4 to i8*
  %sunkaddr16 = getelementptr i8, i8* %8, i64 %sunkaddr
  %9 = bitcast i8* %sunkaddr16 to i64*
  store i64 %acc.2.0.lcssa, i64* %9, align 8
  store i8* null, i8** %retptr, align 8
  ret i32 0
}
